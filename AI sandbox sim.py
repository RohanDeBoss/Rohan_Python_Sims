import pygame
import numpy as np
import random
from collections import deque
import copy

# Define global variable for the number of agents
bot_count = 30  # Increased from 5 to 20
tick_rate = 10  # Initial tick rate (FPS)

# Initialize Pygame
pygame.init()
screen = pygame.display.set_mode((1100, 655))
pygame.display.set_caption(f"AI Agents Simulation - {bot_count} Agents")
clock = pygame.time.Clock()

# Colors
WHITE = (255, 255, 255)
GREEN = (0, 200, 0)
RED = (200, 0, 0)
BLUE = (0, 0, 200)
BLACK = (0, 0, 0)
YELLOW = (255, 255, 0)
LIGHT_GRAY = (240, 240, 240)
DARK_GRAY = (50, 50, 50)
GRAY = (150, 150, 150)

# Configuration
config = {
    'N': 10,
    'num_resources': 20,
    'num_obstacles': 10,
    'hidden_layers': [8],
    'learning_rate': 0.01,
    'gamma': 0.95,
    'epsilon': 1.0,
    'epsilon_decay': 0.99,
    'min_epsilon': 0.05,
    'batch_size': 32,
    'replay_capacity': 5000,
    'max_steps': 100,
    'num_agents': bot_count
}

# Grid display parameters
cell_size = 40
grid_width = config['N'] * cell_size
grid_height = config['N'] * cell_size

# UI Layout
metrics_x = grid_width + 80
metrics_y = 20
nn_x = grid_width + 80
nn_y = 250
graph_x = 60
graph_y = grid_height + 50
graph_width = grid_width
graph_height = 140
controls_x = grid_width + 80
controls_y = graph_y + graph_height - 110
slider_x = grid_width + 80
slider_y = controls_y + 140  # Positioned below controls
slider_width = 200
slider_height = 20
fps_x = 1050  # Top right corner (screen width - padding)
fps_y = 10

# Slider parameters
min_tick_rate = 1    # Minimum FPS
max_tick_rate = 60   # Maximum FPS
slider_range = max_tick_rate - min_tick_rate
slider_pos = (tick_rate - min_tick_rate) / slider_range  # Initial position (0 to 1)

# FPS counter variables
fps_history = deque(maxlen=6)  # Store last 6 FPS values
last_time = pygame.time.get_ticks()  # Time of the last frame

class Environment:
    def __init__(self, N, num_resources, num_obstacles):
        self.N = N
        self.grid = np.zeros((N, N), dtype=int)
        self.agent_pos = [random.randint(0, N-1), random.randint(0, N-1)]
        self.steps = 0
        positions = random.sample([(i, j) for i in range(N) for j in range(N) if (i, j) != tuple(self.agent_pos)], num_resources + num_obstacles)
        for x, y in positions[:num_resources]:
            self.grid[x, y] = 1
        for x, y in positions[num_resources:]:
            self.grid[x, y] = 2

    def reset(self):
        self.__init__(self.N, config['num_resources'], config['num_obstacles'])
        return self.get_input_state()

    def get_state(self):
        x, y = self.agent_pos
        state = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                xi, yj = x + i, y + j
                if 0 <= xi < self.N and 0 <= yj < self.N:
                    state.append(self.grid[xi, yj])
                else:
                    state.append(2)
        return state

    def get_input_state(self):
        state = self.get_state()
        input_vector = []
        for s in state:
            one_hot = [0, 0, 0]
            one_hot[s] = 1
            input_vector.extend(one_hot)
        return np.array(input_vector)

    def step(self, action):
        directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]
        dx, dy = directions[action]
        new_x, new_y = self.agent_pos[0] + dx, self.agent_pos[1] + dy
        reward = 0
        action_result = "Moving"
        if 0 <= new_x < self.N and 0 <= new_y < self.N:
            if self.grid[new_x, new_y] == 1:
                reward = 1
                self.grid[new_x, new_y] = 0
                action_result = "Collected"
            elif self.grid[new_x, new_y] == 2:
                reward = -1
                action_result = "Hit Obstacle"
            else:
                self.agent_pos = [new_x, new_y]
        else:
            reward = -1
            action_result = "Hit Wall"
        if reward == 0 and action_result == "Moving":
            self.agent_pos = [new_x, new_y]
        self.steps += 1
        done = self.steps >= config['max_steps']
        return self.get_input_state(), reward, done, action_result

    def save_state(self):
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos.copy(),
            'steps': self.steps
        }

    def load_state(self, state):
        self.grid = state['grid'].copy()
        self.agent_pos = state['agent_pos'].copy()
        self.steps = state['steps']

class NeuralNetwork:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate):
        self.W1 = np.random.randn(hidden_sizes[0], input_size) * 0.01
        self.b1 = np.zeros(hidden_sizes[0])
        self.W2 = np.random.randn(output_size, hidden_sizes[0]) * 0.01
        self.b2 = np.zeros(output_size)
        self.learning_rate = learning_rate

    def predict(self, input):
        Z1 = np.dot(self.W1, input) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        return Z2, A1

    def train(self, input, target):
        Z1 = np.dot(self.W1, input) + self.b1
        A1 = np.maximum(0, Z1)
        Z2 = np.dot(self.W2, A1) + self.b2
        output = Z2
        dL_dZ2 = output - target
        dL_dW2 = np.outer(dL_dZ2, A1)
        dL_db2 = dL_dZ2
        dL_dA1 = np.dot(self.W2.T, dL_dZ2)
        dL_dZ1 = dL_dA1 * (Z1 > 0)
        dL_dW1 = np.outer(dL_dZ1, input)
        dL_db1 = dL_dZ1
        self.W1 -= self.learning_rate * dL_dW1
        self.b1 -= self.learning_rate * dL_db1
        self.W2 -= self.learning_rate * dL_dW2
        self.b2 -= self.learning_rate * dL_db2
        return A1

class Agent:
    def __init__(self, input_size, hidden_sizes, output_size, learning_rate, gamma, epsilon, epsilon_decay, min_epsilon, batch_size, replay_capacity):
        self.network = NeuralNetwork(input_size, hidden_sizes, output_size, learning_rate)
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.min_epsilon = min_epsilon
        self.batch_size = batch_size
        self.replay_buffer = deque(maxlen=replay_capacity)
        self.action_list = []
        self.initial_env_state = None

    def choose_action(self, state):
        if random.random() < self.epsilon:
            return random.randint(0, 3)
        Q_values, _ = self.network.predict(state)
        return np.argmax(Q_values)

    def store_experience(self, experience):
        self.replay_buffer.append(experience)

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None
        batch = random.sample(self.replay_buffer, self.batch_size)
        hidden_activations = None
        for state, action, reward, new_state, done in batch:
            target = reward if done else reward + self.gamma * np.max(self.network.predict(new_state)[0])
            Q_values, _ = self.network.predict(state)
            Q_values[action] = target
            hidden_activations = self.network.train(state, Q_values)
        if self.epsilon > self.min_epsilon:
            self.epsilon *= self.epsilon_decay
        return hidden_activations

# Drawing functions
def draw_grid(env):
    pygame.draw.rect(screen, DARK_GRAY, (0, 0, grid_width, grid_height), 2)
    for i in range(config['N']):
        for j in range(config['N']):
            rect = pygame.Rect(j * cell_size, i * cell_size, cell_size, cell_size)
            pygame.draw.rect(screen, LIGHT_GRAY, rect)
            if env.grid[i, j] == 1:
                pygame.draw.rect(screen, GREEN, rect.inflate(-10, -10))
            elif env.grid[i, j] == 2:
                points = [(j*cell_size + cell_size//2, i*cell_size + 5),
                          (j*cell_size + 5, i*cell_size + cell_size - 5),
                          (j*cell_size + cell_size - 5, i*cell_size + cell_size - 5)]
                pygame.draw.polygon(screen, RED, points)
    ax, ay = env.agent_pos
    agent_center = (ay * cell_size + cell_size // 2, ax * cell_size + cell_size // 2)
    pygame.draw.circle(screen, BLUE, agent_center, cell_size // 3)

def draw_nn(state, hidden_activations, q_values, action):
    font = pygame.font.SysFont("Arial", 16, bold=True)
    pygame.draw.rect(screen, WHITE, (nn_x - 5, nn_y - 25, 150, 250), 0, 5)
    pygame.draw.rect(screen, DARK_GRAY, (nn_x - 5, nn_y - 25, 150, 250), 2, 5)
    
    if view_mode == "Best Last Gen":
        text = font.render("Brain (Best Ever)", True, BLACK)
    elif view_mode == "Random Agent":
        text = font.render(f"Brain (Agent {random_agent_id})", True, BLACK)
    else:
        text = font.render(f"Brain (Agent {best_agent_id})", True, BLACK)
    screen.blit(text, (nn_x, nn_y - 20))
    
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            s = state[idx * 3: (idx + 1) * 3]
            color = WHITE if s[0] == 1 else GREEN if s[1] == 1 else RED
            rect = pygame.Rect(nn_x + j * 15, nn_y + i * 15, 15, 15)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)

    if hidden_activations is not None:
        max_act = max(hidden_activations) or 1
        for i, act in enumerate(hidden_activations):
            height = int((act / max_act) * 30)
            rect = pygame.Rect(nn_x + i * 10, nn_y + 100 - height, 8, height)
            pygame.draw.rect(screen, BLUE, rect)

    for i, q in enumerate(q_values):
        height = int((q + 1) * 20)
        rect = pygame.Rect(nn_x + i * 20, nn_y + 210 - height, 15, height)
        color = GREEN if i == action else GRAY
        pygame.draw.rect(screen, color, rect)

def draw_nn_at(state, hidden_activations, q_values, action, x, y, label):
    title_font = pygame.font.SysFont("Arial", 20, bold=True)
    label_font = pygame.font.SysFont("Arial", 16, bold=True)
    small_font = pygame.font.SysFont("Arial", 12)
    pygame.draw.rect(screen, WHITE, (x - 5, y - 30, 150, 310), 0, 5)
    pygame.draw.rect(screen, DARK_GRAY, (x - 5, y - 30, 150, 310), 2, 5)
    text = title_font.render(label, True, BLACK)
    screen.blit(text, (x, y - 25))
    text = label_font.render("Input (3x3)", True, BLACK)
    screen.blit(text, (x, y))
    for i in range(3):
        for j in range(3):
            idx = i * 3 + j
            s = state[idx * 3: (idx + 1) * 3]
            color = WHITE if s[0] == 1 else GREEN if s[1] == 1 else RED
            rect = pygame.Rect(x + j * 15, y + 20 + i * 15, 15, 15)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, BLACK, rect, 1)
    text = label_font.render("Hidden Layer", True, BLACK)
    screen.blit(text, (x, y + 70))
    if hidden_activations is not None:
        max_act = max(hidden_activations) or 1
        for i, act in enumerate(hidden_activations):
            height = int((act / max_act) * 30)
            rect = pygame.Rect(x + i * 10, y + 120 - height, 8, height)
            pygame.draw.rect(screen, BLUE, rect)
    text = label_font.render("Q-Values", True, BLACK)
    screen.blit(text, (x, y + 120))
    for i, q in enumerate(q_values):
        height = int((q + 1) * 20)
        rect = pygame.Rect(x + i * 20, y + 253 - height, 15, height)
        color = GREEN if i == action else GRAY
        pygame.draw.rect(screen, color, rect)
    action_labels = ["Up", "Down", "   Left", "   Right"]
    for i, label in enumerate(action_labels):
        text = small_font.render(label, True, BLACK)
        screen.blit(text, (x + i * 18, y + 260))

def draw_metrics(episode, avg_reward, best_reward, avg_epsilon, best_agent_id, action_result, view_mode_text):
    font = pygame.font.SysFont("Arial", 20, bold=True)
    pygame.draw.rect(screen, WHITE, (metrics_x - 5, metrics_y - 5, 280, 200), 0, 5)
    pygame.draw.rect(screen, DARK_GRAY, (metrics_x - 5, metrics_y - 5, 280, 200), 2, 5)
    metrics = [
        f"Generation: {episode + 1}",
        f"Avg Reward: {avg_reward:.1f}",
        f"Best: {best_reward:.1f} (A{best_agent_id})",
        f"Epsilon: {avg_epsilon:.2f}",
        f"Action: {action_result}",
        f"Viewing: {view_mode_text}"
    ]
    for i, metric in enumerate(metrics):
        text = font.render(metric, True, BLACK)
        screen.blit(text, (metrics_x, metrics_y + i * 30))

def draw_graph(avg_reward_history, best_reward_history):
    if len(avg_reward_history) < 2:
        return
    pygame.draw.rect(screen, WHITE, (graph_x - 5, graph_y - 5, graph_width + 10, graph_height + 40), 0, 5)
    pygame.draw.rect(screen, DARK_GRAY, (graph_x - 5, graph_y - 5, graph_width + 10, graph_height + 40), 2, 5)
    font = pygame.font.SysFont("Arial", 16, bold=True)
    text = font.render("Learning Progress", True, BLACK)
    screen.blit(text, (graph_x + graph_width // 2 - 50, graph_y - 30))
    
    max_reward = max(max(avg_reward_history), max(best_reward_history)) or 1
    min_reward = min(min(avg_reward_history), min(best_reward_history)) or -1
    scale_y = (graph_height - 10) / (max_reward - min_reward) if max_reward != min_reward else 1
    
    if len(avg_reward_history) <= 50:
        points_to_plot_avg = avg_reward_history
        points_to_plot_best = best_reward_history
        num_points = len(avg_reward_history)
        if num_points > 1:
            scale_x = graph_width / (num_points - 1)
        else:
            scale_x = 0
        start_gen = 1
    else:
        points_to_plot_avg = avg_reward_history[-50:]
        points_to_plot_best = best_reward_history[-50:]
        num_points = 50
        scale_x = graph_width / 49
        start_gen = len(avg_reward_history) - 49
    
    avg_points = []
    for i in range(num_points):
        reward = points_to_plot_avg[i]
        x = graph_x + i * scale_x
        y = graph_y + graph_height - 5 - (reward - min_reward) * scale_y
        avg_points.append((x, y))
    pygame.draw.lines(screen, YELLOW, False, avg_points, 2)
    
    best_points = []
    for i in range(num_points):
        reward = points_to_plot_best[i]
        x = graph_x + i * scale_x
        y = graph_y + graph_height - 5 - (reward - min_reward) * scale_y
        best_points.append((x, y))
    pygame.draw.lines(screen, BLUE, False, best_points, 2)
    
    font = pygame.font.SysFont("Arial", 14)
    y_label = font.render("Reward", True, BLACK)
    screen.blit(y_label, (graph_x - 50, graph_y + graph_height // 2 - 10))
    min_reward_text = font.render(f"{min_reward:.1f}", True, BLACK)
    screen.blit(min_reward_text, (graph_x - 40, graph_y + graph_height - 15))
    max_reward_text = font.render(f"{max_reward:.1f}", True, BLACK)
    screen.blit(max_reward_text, (graph_x - 40, graph_y))
    
    x_label = font.render("Generation", True, BLACK)
    screen.blit(x_label, (graph_x + graph_width // 2 - 30, graph_y + graph_height + 10))
    
    for idx in [0, 10, 20, 30, 40]:
        if idx < num_points:
            gen = start_gen + idx
            x = graph_x + idx * scale_x
            if x < graph_x + graph_width:
                pygame.draw.line(screen, BLACK, (x, graph_y + graph_height - 5), (x, graph_y + graph_height), 2)
                text = font.render(str(gen), True, BLACK)
                screen.blit(text, (x - 10, graph_y + graph_height + 5))

    pygame.draw.line(screen, YELLOW, (graph_x + 10, graph_y - 32), (graph_x + 30, graph_y - 32), 2)
    screen.blit(font.render("Avg Reward", True, BLACK), (graph_x + 35, graph_y - 37))
    pygame.draw.line(screen, BLUE, (graph_x + 10, graph_y - 17), (graph_x + 30, graph_y - 17), 2)
    screen.blit(font.render("Best Reward", True, BLACK), (graph_x + 35, graph_y - 22))

def draw_controls():
    font = pygame.font.SysFont("Arial", 16, bold=True)
    controls = ["Space: Pause/Resume", "S: Step", "R: Reset", "B: Best Last Gen", "V: Random Agent", "C: Best Current"]
    for i, control in enumerate(controls):
        text = font.render(control, True, BLACK)
        screen.blit(text, (controls_x, controls_y + i * 20))

def draw_slider():
    global tick_rate, slider_pos
    font = pygame.font.SysFont("Arial", 16, bold=True)
    
    pygame.draw.rect(screen, GRAY, (slider_x, slider_y, slider_width, slider_height))
    pygame.draw.rect(screen, DARK_GRAY, (slider_x, slider_y, slider_width, slider_height), 2)
    
    handle_x = slider_x + int(slider_pos * (slider_width - 20))
    handle_rect = pygame.Rect(handle_x, slider_y - 5, 20, slider_height + 10)
    pygame.draw.rect(screen, BLUE, handle_rect)
    pygame.draw.rect(screen, BLACK, handle_rect, 2)
    
    label = font.render("Sim Speed:", True, BLACK)
    screen.blit(label, (slider_x, slider_y - 25))
    value_text = font.render(f"{int(tick_rate)} FPS", True, BLACK)
    screen.blit(value_text, (slider_x + 100, slider_y - 25))

    mouse_pos = pygame.mouse.get_pos()
    mouse_pressed = pygame.mouse.get_pressed()[0]
    
    if mouse_pressed:
        if handle_rect.collidepoint(mouse_pos):
            new_pos = (mouse_pos[0] - slider_x - 10) / (slider_width - 20)
            slider_pos = max(0, min(1, new_pos))
            tick_rate = min_tick_rate + slider_pos * slider_range
            tick_rate = max(min_tick_rate, min(max_tick_rate, tick_rate))

def draw_fps_counter():
    global fps_history, last_time
    font = pygame.font.SysFont("Arial", 16, bold=True)
    
    # Calculate current FPS
    current_time = pygame.time.get_ticks()
    delta_time = current_time - last_time
    if delta_time > 0:  # Prevent division by zero
        current_fps = 1000 / delta_time  # Convert milliseconds to FPS
        fps_history.append(current_fps)
    last_time = current_time
    
    # Calculate average FPS
    if len(fps_history) > 0:
        avg_fps = sum(fps_history) / len(fps_history)
        text = font.render(f"FPS: {avg_fps:.1f}", True, BLACK)
        text_rect = text.get_rect(topright=(fps_x, fps_y))
        pygame.draw.rect(screen, WHITE, text_rect.inflate(4, 4))  # Background
        pygame.draw.rect(screen, DARK_GRAY, text_rect.inflate(4, 4), 1)  # Border
        screen.blit(text, text_rect)

# Initialize agents and environments
envs = [Environment(config['N'], config['num_resources'], config['num_obstacles']) for _ in range(bot_count)]
agents = [Agent(27, config['hidden_layers'], 4, config['learning_rate'], config['gamma'], config['epsilon'],
                config['epsilon_decay'], config['min_epsilon'], config['batch_size'], config['replay_capacity'])
          for _ in range(bot_count)]

# Best agent from previous generation and replay variables
best_agent_prev = None
best_reward_prev = -float('inf')
best_initial_env_state = None
best_actions_prev = []
best_agent_prev_id = None

# Replay state variables
replay_active = False
replay_step = 0
replay_env = None
replay_actions = []

# Game state
running = True
paused = False
step_mode = False
episode = 0
total_rewards = [0] * bot_count
avg_reward_history = []
best_reward_history = []
action_result = "Starting"
view_mode = "Best Current"
random_agent_id = random.randint(0, bot_count - 1)
best_agent_id = 0

# Main game loop
while running:
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                paused = not paused
            elif event.key == pygame.K_s:
                step_mode = True
                paused = False
            elif event.key == pygame.K_r:
                envs = [Environment(config['N'], config['num_resources'], config['num_obstacles']) for _ in range(bot_count)]
                agents = [Agent(27, config['hidden_layers'], 4, config['learning_rate'], config['gamma'], config['epsilon'],
                                config['epsilon_decay'], config['min_epsilon'], config['batch_size'], config['replay_capacity'])
                          for _ in range(bot_count)]
                total_rewards = [0] * bot_count
                avg_reward_history = []
                best_reward_history = []
                best_agent_prev = None
                best_initial_env_state = None
                best_actions_prev = []
                best_agent_prev_id = None
                episode = 0
                replay_active = False
                view_mode = "Best Current"
            elif event.key == pygame.K_b:
                if best_agent_prev:
                    view_mode = "Best Last Gen"
                    replay_active = True
                    replay_step = 0
                    replay_env = Environment(config['N'], 0, 0)
                    replay_env.load_state(best_initial_env_state)
                    replay_actions = best_actions_prev
            elif event.key == pygame.K_v:
                view_mode = "Random Agent"
                random_agent_id = random.randint(0, bot_count - 1)
                replay_active = False
            elif event.key == pygame.K_c:
                view_mode = "Best Current"
                replay_active = False

    screen.fill(LIGHT_GRAY)

    if replay_active:
        if step_mode or not paused:
            if replay_step < len(replay_actions):
                action = replay_actions[replay_step]
                _, reward, done, action_result = replay_env.step(action)
                replay_step += 1
        current_env = replay_env
        current_agent = best_agent_prev
        view_mode_text = f"Best Last Gen (Reward: {best_reward_prev})"
    else:
        if not paused or step_mode:
            done_any = False
            best_agent_id = np.argmax(total_rewards)
            for i in range(bot_count):
                if envs[i].steps == 0:
                    agents[i].initial_env_state = envs[i].save_state()
                    agents[i].action_list = []
                state = envs[i].get_input_state()
                action = agents[i].choose_action(state)
                new_state, reward, done, action_result_i = envs[i].step(action)
                agents[i].store_experience((state, action, reward, new_state, done))
                agents[i].train()
                total_rewards[i] += reward
                agents[i].action_list.append(action)
                if done:
                    done_any = True

            avg_reward = sum(total_rewards) / bot_count

            if view_mode == "Random Agent":
                current_env = envs[random_agent_id]
                current_agent = agents[random_agent_id]
                view_mode_text = "Random Agent"
                action_result = "N/A"
            else:
                current_env = envs[best_agent_id]
                current_agent = agents[best_agent_id]
                view_mode_text = "Best Current"
                action_result = "N/A"

    draw_grid(current_env)
    q_values, hidden_activations = current_agent.network.predict(current_env.get_input_state())
    action = np.argmax(q_values)
    draw_nn(current_env.get_input_state(), hidden_activations, q_values, action)
    draw_metrics(episode, sum(total_rewards) / bot_count, max(total_rewards),
                 sum(agent.epsilon for agent in agents) / bot_count,
                 best_agent_id, action_result, view_mode_text)
    draw_graph(avg_reward_history, best_reward_history)
    draw_controls()
    draw_slider()
    draw_fps_counter()  # Add FPS counter drawing

    best_state = envs[best_agent_id].get_input_state()
    best_q_values, best_hidden_activations = agents[best_agent_id].network.predict(best_state)
    best_action = np.argmax(best_q_values)
    draw_nn_at(best_state, best_hidden_activations, best_q_values, best_action, 780, 100, "Best Agent NN")

    pygame.display.flip()

    if not paused or step_mode:
        if done_any:
            episode += 1
            avg_reward_history.append(avg_reward)
            best_reward_history.append(max(total_rewards))
            if max(total_rewards) > best_reward_prev:
                best_reward_prev = max(total_rewards)
                best_agent_prev = copy.deepcopy(agents[best_agent_id])
                best_initial_env_state = agents[best_agent_id].initial_env_state
                best_actions_prev = agents[best_agent_id].action_list.copy()
                best_agent_prev_id = best_agent_id
            total_rewards = [0] * bot_count
            for env in envs:
                env.reset()

        step_mode = False

    clock.tick(tick_rate)

pygame.quit()