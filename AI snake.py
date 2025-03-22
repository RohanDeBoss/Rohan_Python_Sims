import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import threading
import time
from collections import deque

# --------------------- Constants ---------------------
GRID_SIZE = 20
CELL_SIZE = 20
SIM_WIDTH = GRID_SIZE * CELL_SIZE   # 400 pixels
SIM_HEIGHT = GRID_SIZE * CELL_SIZE  # 400 pixels

WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 670

LEFT_PANEL_WIDTH = 960
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - LEFT_PANEL_WIDTH

CONTROL_AREA_HEIGHT = 200
STATS_AREA_HEIGHT = WINDOW_HEIGHT - CONTROL_AREA_HEIGHT

NUM_AGENTS = 30
BATCH_SIZE = 32
REPLAY_BUFFER_SIZE = 10000
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 10000
GAMMA = 0.99
LEARNING_RATE = 0.001
TARGET_UPDATE_FREQ = 1000

MIN_SPEED_MULTIPLIER = 1
MAX_SPEED_MULTIPLIER = 10
DEFAULT_SPEED_MULTIPLIER = 1

DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)]

BACKGROUND_COLOR = (10, 10, 20)
GRID_COLOR = (40, 40, 70)
SNAKE_HEAD_COLOR = (0, 255, 100)
SNAKE_BODY_COLOR = (0, 200, 70)
FOOD_COLOR = (255, 50, 50)
TEXT_COLOR = (220, 220, 220)
GRAPH_BACKGROUND = (20, 20, 40)
GRAPH_LINE_COLOR = (50, 200, 100)
GRAPH_GRID_COLOR = (50, 50, 80)

# --------------------- Global Variables ---------------------
speed_multiplier = DEFAULT_SPEED_MULTIPLIER
training_paused = False

# --------------------- Game Classes ---------------------
class SnakeGame:
    def __init__(self):
        self.reset()

    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]
        self.direction = 0
        self.food = self._spawn_food()
        self.score = 0
        self.done = False
        self.steps = 0
        self.max_score = 0
        return self._get_state()

    def _spawn_food(self):
        while True:
            food = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if food not in self.snake:
                return food

    def step(self, action):
        if self.done:
            return self._get_state(), 0, True

        self.steps += 1
        new_dir = (self.direction + (action - 1)) % 4
        dx, dy = DIRECTIONS[new_dir]
        head_x, head_y = self.snake[0]
        new_head = (head_x + dx, head_y + dy)

        if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
            new_head[1] < 0 or new_head[1] >= GRID_SIZE or
            new_head in self.snake):
            self.done = True
            return self._get_state(), -1, True

        self.snake.insert(0, new_head)
        if new_head == self.food:
            self.score += 1
            if self.score > self.max_score:
                self.max_score = self.score
            self.food = self._spawn_food()
            reward = 1
        else:
            self.snake.pop()
            reward = -0.01

        self.direction = new_dir
        return self._get_state(), reward, self.done

    def _get_state(self):
        head_x, head_y = self.snake[0]
        food_x, food_y = self.food
        dir_onehot = [1 if i == self.direction else 0 for i in range(4)]
        delta_x = (food_x - head_x) / GRID_SIZE
        delta_y = (food_y - head_y) / GRID_SIZE
        dangers = []
        for turn in [-1, 0, 1]:
            new_dir = (self.direction + turn) % 4
            dx, dy = DIRECTIONS[new_dir]
            new_head = (head_x + dx, head_y + dy)
            danger = 1 if (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                           new_head[1] < 0 or new_head[1] >= GRID_SIZE or
                           new_head in self.snake) else 0
            dangers.append(danger)
        return np.array(dir_onehot + [delta_x, delta_y] + dangers, dtype=np.float32)

    def render(self, surface):
        surface.fill(BACKGROUND_COLOR)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE):
                rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
                pygame.draw.rect(surface, GRID_COLOR, rect, 1)
        for i, (x, y) in enumerate(self.snake):
            rect = pygame.Rect(x * CELL_SIZE, y * CELL_SIZE, CELL_SIZE, CELL_SIZE)
            color = SNAKE_HEAD_COLOR if i == 0 else SNAKE_BODY_COLOR
            pygame.draw.rect(surface, color, rect)
        food_rect = pygame.Rect(self.food[0] * CELL_SIZE, self.food[1] * CELL_SIZE, CELL_SIZE, CELL_SIZE)
        pygame.draw.rect(surface, FOOD_COLOR, food_rect)

# --------------------- Replay Buffer ---------------------
class ReplayBuffer:
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)
    def add(self, experience):
        self.buffer.append(experience)
    def sample(self, batch_size):
        return random.sample(self.buffer, batch_size)
    def __len__(self):
        return len(self.buffer)

# --------------------- Neural Network ---------------------
class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.layer_sizes = [input_dim, 64, 64, output_dim]
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(),
            nn.Linear(64, 64),
            nn.ReLU(),
            nn.Linear(64, output_dim)
        )
    def forward(self, x):
        return self.net(x)
    def get_weights(self):
        weights = []
        for name, param in self.named_parameters():
            if 'weight' in name:
                weights.append(param.detach().numpy())
        return weights

# --------------------- Agent ---------------------
class DQNAgent:
    def __init__(self, state_dim, action_dim):
        self.policy_net = DQN(state_dim, action_dim)
        self.target_net = DQN(state_dim, action_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=LEARNING_RATE)
        self.replay_buffer = ReplayBuffer(REPLAY_BUFFER_SIZE)
        self.steps = 0
        self.losses = deque(maxlen=100)
        self.episodes = 0
        self.total_reward = 0

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, 2)
        with torch.no_grad():
            q_values = self.policy_net(torch.tensor(state, dtype=torch.float32))
            return q_values.argmax().item()

    def store_experience(self, experience):
        state, action, reward, next_state, done = experience
        self.replay_buffer.add(experience)
        self.total_reward += reward
        if done:
            self.episodes += 1
            self.total_reward = 0

    def update_network(self):
        if len(self.replay_buffer) < BATCH_SIZE:
            return
        batch = self.replay_buffer.sample(BATCH_SIZE)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.tensor(np.array(states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.long)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        current_q = self.policy_net(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            next_q = self.target_net(next_states).max(1)[0]
            target_q = rewards + GAMMA * next_q * (1 - dones)
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        self.losses.append(loss.item())
        self.steps += 1
        if self.steps % TARGET_UPDATE_FREQ == 0:
            self.target_net.load_state_dict(self.policy_net.state_dict())

# --------------------- Stats & Visualization ---------------------
class StatsTracker:
    def __init__(self, num_agents):
        self.num_agents = num_agents
        self.avg_scores = []
        self.current_max_scores = []
        self.training_steps = 0

    def update(self, games, agents):
        self.training_steps += 1
        if self.training_steps % 10 == 0:
            avg_score = np.mean([game.score for game in games])
            current_max_score = max([game.score for game in games])
            self.avg_scores.append(avg_score)
            self.current_max_scores.append(current_max_score)

    def get_surface(self, width, height, games, agents):
        surface = pygame.Surface((width, height))
        surface.fill(GRAPH_BACKGROUND)
        font = pygame.font.Font(None, 20)
        
        if len(self.avg_scores) < 2:
            msg = font.render(f"Gathering training data... ({len(self.avg_scores)} data points)", True, TEXT_COLOR)
            surface.blit(msg, (width//2 - msg.get_width()//2, height//2 - msg.get_height()//2))
            return surface

        plot_area = pygame.Rect(50, 70, width - 70, 300)

        def plot_line(data, color):
            pts = []
            max_val = max(max(self.avg_scores), max(self.current_max_scores)) if self.avg_scores and self.current_max_scores else 1
            for i, val in enumerate(data):
                x = plot_area.x + int(i / (len(data) - 1) * plot_area.width) if len(data) > 1 else plot_area.x
                y = plot_area.bottom - int((val / max_val) * plot_area.height)
                pts.append((x, y))
            if len(pts) > 1:
                pygame.draw.lines(surface, color, False, pts, 2)

        plot_line(self.avg_scores, GRAPH_LINE_COLOR)
        plot_line(self.current_max_scores, (255, 255, 100))

        # Draw axes
        pygame.draw.line(surface, TEXT_COLOR, (plot_area.x, plot_area.bottom), (plot_area.x, plot_area.y), 1)  # Y-axis
        pygame.draw.line(surface, TEXT_COLOR, (plot_area.x, plot_area.bottom), (plot_area.right, plot_area.bottom), 1)  # X-axis

        # Y-axis markers (scores)
        max_val = max(max(self.avg_scores), max(self.current_max_scores)) if self.avg_scores and self.current_max_scores else 1
        step_size = max(1, int(max_val / 5))
        for val in range(0, int(max_val) + 1, step_size):
            y = plot_area.bottom - int((val / max_val) * plot_area.height)
            pygame.draw.line(surface, TEXT_COLOR, (plot_area.x - 5, y), (plot_area.x, y), 1)
            label = font.render(str(val), True, TEXT_COLOR)
            surface.blit(label, (plot_area.x - 30, y - label.get_height() // 2))

        # X-axis markers (generation number)
        total_episodes = sum(agent.episodes for agent in agents)
        num_points = len(self.avg_scores)
        episodes_per_point = total_episodes / num_points if num_points > 0 else 1
        step_interval = max(1, num_points // 5)
        for i in range(0, num_points, step_interval):
            x = plot_area.x + int(i / (num_points - 1) * plot_area.width) if num_points > 1 else plot_area.x
            pygame.draw.line(surface, TEXT_COLOR, (x, plot_area.bottom), (x, plot_area.bottom + 5), 1)
            gen_num = int(i * episodes_per_point)
            label = font.render(str(gen_num), True, TEXT_COLOR)
            surface.blit(label, (x - label.get_width() // 2, plot_area.bottom + 5))

        # Labels
        avg_label = font.render("Avg Score", True, GRAPH_LINE_COLOR)
        surface.blit(avg_label, (width - avg_label.get_width() - 10, 10))
        max_label = font.render("Best Score", True, (255, 255, 100))
        surface.blit(max_label, (width - max_label.get_width() - 10, 30))

        # Current values
        current_avg_score = np.mean([game.score for game in games])
        current_best_score = max([game.score for game in games])
        current_avg_text = font.render(f"Current Avg: {current_avg_score:.2f}", True, GRAPH_LINE_COLOR)
        current_best_text = font.render(f"Current Best: {current_best_score}", True, (255, 255, 100))
        surface.blit(current_avg_text, (10, 10))
        surface.blit(current_best_text, (10, 30))

        return surface

def visualize_network(agent, surface):
    surface.fill(GRAPH_BACKGROUND)
    try:
        weights = agent.policy_net.get_weights()
        layers = agent.policy_net.layer_sizes
        padding = 20
        layer_width = (surface.get_width() - 2 * padding) / (len(layers) - 1)
        max_neurons = max(layers)
        neuron_radius = max(4, int((surface.get_height() - 2 * padding) / (2 * max_neurons)))
        positions = []
        for i, n in enumerate(layers):
            x = padding + i * layer_width
            y_spacing = (surface.get_height() - 2 * padding) / n
            layer_positions = [(int(x), int(padding + y_spacing * (j + 0.5))) for j in range(n)]
            positions.append(layer_positions)
            for pos in layer_positions:
                pygame.draw.circle(surface, TEXT_COLOR, pos, neuron_radius)
        for i in range(len(layers)-1):
            weight_matrix = weights[i]
            norm = np.max(np.abs(weight_matrix)) + 1e-5
            for j, pos1 in enumerate(positions[i]):
                for k, pos2 in enumerate(positions[i+1]):
                    if layers[i]*layers[i+1] > 100 and random.random() > 0.3:
                        continue
                    weight = weight_matrix[k, j] / norm
                    color = (0, min(255, int(255 * weight + 128)), 0) if weight > 0 else (min(255, int(255 * abs(weight) + 128)), 0, 0)
                    pygame.draw.line(surface, color, pos1, pos2, max(1, int(2 * abs(weight))))
        font = pygame.font.Font(None, 24)
        title = font.render("Neural Network", True, TEXT_COLOR)
        surface.blit(title, (surface.get_width()//2 - title.get_width()//2, 5))
    except Exception as e:
        font = pygame.font.Font(None, 24)
        err = font.render("NN visualization error", True, (255, 100, 100))
        surface.blit(err, (surface.get_width()//2 - err.get_width()//2, surface.get_height()//2))
    return surface

# --------------------- Training Loop ---------------------
def training_loop(games, agents, stats_tracker, stop_event):
    global speed_multiplier, training_paused
    while not stop_event.is_set():
        if not training_paused:
            for _ in range(speed_multiplier):
                for game, agent in zip(games, agents):
                    state = game._get_state()
                    epsilon = max(EPSILON_END, EPSILON_START - (EPSILON_START - EPSILON_END) * (agent.steps / EPSILON_DECAY))
                    action = agent.select_action(state, epsilon)
                    next_state, reward, done = game.step(action)
                    agent.store_experience((state, action, reward, next_state, done))
                    agent.update_network()
                    if done:
                        game.reset()
                stats_tracker.update(games, agents)
        time.sleep(0.001)

# --------------------- Main Simulation ---------------------
def main():
    global speed_multiplier, training_paused
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Enhanced AI Snake Simulation")
    clock = pygame.time.Clock()

    games = [SnakeGame() for _ in range(NUM_AGENTS)]
    agents = [DQNAgent(9, 3) for _ in range(NUM_AGENTS)]
    stats_tracker = StatsTracker(NUM_AGENTS)

    left_panel = pygame.Rect(0, 0, LEFT_PANEL_WIDTH, WINDOW_HEIGHT)
    control_area = pygame.Rect(LEFT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, CONTROL_AREA_HEIGHT)
    stats_area = pygame.Rect(LEFT_PANEL_WIDTH, CONTROL_AREA_HEIGHT, RIGHT_PANEL_WIDTH, STATS_AREA_HEIGHT)

    simulation_surface = pygame.Surface((SIM_WIDTH, SIM_HEIGHT))

    stop_event = threading.Event()
    train_thread = threading.Thread(target=training_loop, args=(games, agents, stats_tracker, stop_event))
    train_thread.daemon = True
    train_thread.start()

    selected_agent = 0
    show_nn = False

    small_font = pygame.font.Font(None, 22)
    med_font = pygame.font.Font(None, 28)
    large_font = pygame.font.Font(None, 36)

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    training_paused = not training_paused
                elif event.key == pygame.K_n:
                    show_nn = not show_nn
                elif event.key == pygame.K_b:
                    selected_agent = max(range(NUM_AGENTS), key=lambda i: games[i].score)
                elif event.key == pygame.K_r:
                    selected_agent = random.randint(0, NUM_AGENTS - 1)
                elif event.key == pygame.K_UP:
                    speed_multiplier = min(MAX_SPEED_MULTIPLIER, speed_multiplier + 1)
                elif event.key == pygame.K_DOWN:
                    speed_multiplier = max(MIN_SPEED_MULTIPLIER, speed_multiplier - 1)

        screen.fill(BACKGROUND_COLOR)

        simulation_surface.fill(BACKGROUND_COLOR)
        games[selected_agent].render(simulation_surface)
        sim_x = left_panel.x + (left_panel.width - SIM_WIDTH) // 2
        sim_y = left_panel.y + (left_panel.height - SIM_HEIGHT) // 2
        screen.blit(simulation_surface, (sim_x, sim_y))

        pygame.draw.rect(screen, GRAPH_BACKGROUND, control_area)
        status_text = "PAUSED" if training_paused else "TRAINING"
        status_color = (200, 50, 50) if training_paused else (50, 200, 50)
        status_surf = large_font.render(status_text, True, status_color)
        screen.blit(status_surf, (control_area.x + 10, control_area.y + 10))

        controls = [
            "SPACE: Pause/Resume",
            f"UP/DOWN: Speed ({speed_multiplier}x)",
            "B: Best Snake",
            "R: Random Snake",
            "N: Toggle NN View"
        ]
        for i, text in enumerate(controls):
            ctrl_surf = small_font.render(text, True, TEXT_COLOR)
            screen.blit(ctrl_surf, (control_area.x + 10, control_area.y + 60 + i * 25))

        agent_game = games[selected_agent]
        agent_obj = agents[selected_agent]
        info = [
            f"Agent {selected_agent}",
            f"Score: {agent_game.score}",
            f"Max Score: {agent_game.max_score}",
            f"Steps: {agent_obj.steps}",
            f"Episodes: {agent_obj.episodes}"
        ]
        for i, line in enumerate(info):
            info_surf = small_font.render(line, True, TEXT_COLOR)
            screen.blit(info_surf, (control_area.x + 10, control_area.y + 180 + i * 20))

        agents_alive = sum(1 for game in games if not game.done)
        total_episodes = sum(agent.episodes for agent in agents)
        global_stats = [
            f"Total Agents: {NUM_AGENTS}",
            f"Agents Alive: {agents_alive}",
            f"Total Episodes: {total_episodes}",
            f"Global Avg Score: {np.mean([g.score for g in games]):.2f}",
            f"Global Max Score: {max([g.max_score for g in games])}"
        ]
        for i, line in enumerate(global_stats):
            gs_surf = small_font.render(line, True, TEXT_COLOR)
            screen.blit(gs_surf, (control_area.x + 10, control_area.y + 300 + i * 20))

        pygame.draw.rect(screen, GRAPH_BACKGROUND, stats_area)
        if show_nn:
            visualize_network(agents[selected_agent], screen.subsurface(stats_area))
        else:
            stats_surface = stats_tracker.get_surface(stats_area.width, stats_area.height, games, agents)
            screen.blit(stats_surface, stats_area)

        pygame.display.flip()
        clock.tick(30)

    stop_event.set()
    train_thread.join(timeout=1.0)
    pygame.quit()

if __name__ == "__main__":
    main()