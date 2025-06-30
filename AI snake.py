import pygame
import random
import numpy as np
import torch
import torch.nn as nn
import time
from collections import deque

# --------------------- Constants ---------------------
# --- Simulation & Display ---
GRID_SIZE = 12
WINDOW_WIDTH = 1280
WINDOW_HEIGHT = 690
LEFT_PANEL_WIDTH = 960
RIGHT_PANEL_WIDTH = WINDOW_WIDTH - LEFT_PANEL_WIDTH
CONTROL_AREA_HEIGHT = 340
STATS_AREA_HEIGHT = WINDOW_HEIGHT - CONTROL_AREA_HEIGHT
MIN_SPEED, MAX_SPEED, DEFAULT_SPEED = 1, 50, 8
SPEED_DIVISOR = 10.0

# --- Genetic Algorithm ---
POPULATION_SIZE = 36
AI_INPUT_SIZE = 7
ELITISM_COUNT = 2
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.2

# --- Game Logic ---
HUNGER_LIMIT = GRID_SIZE * 4
MIN_REPETITION_LIMIT = 4
MAX_REPETITION_LIMIT = 24
DEFAULT_REPETITION_LIMIT = 14

# --- Colors & Fonts ---
BACKGROUND_COLOR = (10, 10, 20); GRID_COLOR = (40, 40, 70)
SNAKE_HEAD_COLOR = (0, 255, 100); SNAKE_BODY_COLOR = (0, 200, 70)
FOOD_COLOR = (255, 50, 50); TEXT_COLOR = (220, 220, 220)
GRAPH_BACKGROUND = (20, 20, 40); GRAPH_LINE_COLOR = (50, 200, 100); GRAPH_MAX_COLOR = (255, 255, 100)
LEADER_COLOR = (50, 150, 255)
DEAD_COLOR = (255, 200, 0, 100); REPETITION_DEAD_COLOR = (255, 0, 0, 100)
EYE_COLOR = (0, 0, 0)
SLIDER_TRACK_COLOR = (40, 40, 70); SLIDER_HANDLE_COLOR = (150, 150, 180)

# --------------------- Game Logic Class ---------------------
class SnakeGame:
    DIRECTIONS = [(0, -1), (1, 0), (0, 1), (-1, 0)] # 0:Up, 1:Right, 2:Down, 3:Left
    def __init__(self): self.reset()
    def reset(self):
        self.snake = [(GRID_SIZE // 2, GRID_SIZE // 2)]; self.direction = random.randint(0, 3)
        self.food = self._spawn_food(); self.score = 0; self.total_steps = 0
        self.steps_since_food = 0; self.is_done = False
        self.death_reason = None
        self.action_history = deque(maxlen=MAX_REPETITION_LIMIT)
    def _spawn_food(self):
        while True:
            pos = (random.randint(0, GRID_SIZE - 1), random.randint(0, GRID_SIZE - 1))
            if pos not in self.snake: return pos

    def step(self, action, repetition_limit):
        if self.is_done: return
        self.steps_since_food += 1; self.total_steps += 1
        self.action_history.append(action)

        # First, update the snake's direction and position
        self.direction = (self.direction + (action - 1)) % 4
        head_x, head_y = self.snake[0]
        dx, dy = self.DIRECTIONS[self.direction]
        new_head = (head_x + dx, head_y + dy)
        self.snake.insert(0, new_head)

        # Now, check for all death conditions based on the new state
        is_out_of_bounds = (new_head[0] < 0 or new_head[0] >= GRID_SIZE or
                            new_head[1] < 0 or new_head[1] >= GRID_SIZE)
        is_self_collision = new_head in self.snake[1:]
        is_starving = self.steps_since_food > HUNGER_LIMIT
        
        effective_limit = min(MAX_REPETITION_LIMIT, repetition_limit + self.score)
        is_repetition = False
        if len(self.action_history) >= effective_limit:
            half = effective_limit // 2
            if half > 0 and list(self.action_history)[-effective_limit:-half] == list(self.action_history)[-half:]:
                is_repetition = True

        if is_out_of_bounds or is_self_collision or is_starving or is_repetition:
            self.is_done = True
            if is_repetition: self.death_reason = 'repetition'
            elif is_starving: self.death_reason = 'starvation'
            else: self.death_reason = 'collision'
            # We don't return here, allowing the invalid state to be rendered once
        
        # If not dead, handle food and tail popping
        if not self.is_done:
            if new_head == self.food:
                self.score += 1
                self.steps_since_food = 0
                self.food = self._spawn_food()
                self.action_history.clear()
            else:
                self.snake.pop()

    def get_state(self):
        head_x, head_y = self.snake[0]; state = []
        for turn in [-1, 0, 1]:
            check_dir = (self.direction + turn) % 4; dx, dy = self.DIRECTIONS[check_dir]; next_pos = (head_x + dx, head_y + dy)
            is_danger = (next_pos in self.snake[:-1] or next_pos[0] < 0 or next_pos[0] >= GRID_SIZE or next_pos[1] < 0 or next_pos[1] >= GRID_SIZE)
            state.append(1.0 if is_danger else 0.0)
        state.extend([1.0 if self.food[1] < head_y else 0.0, 1.0 if self.food[1] > head_y else 0.0, 1.0 if self.food[0] < head_x else 0.0, 1.0 if self.food[0] > head_x else 0.0])
        return torch.FloatTensor(state)

    def render(self, surface, cell_size, head_surface):
        surface.fill(BACKGROUND_COLOR)
        for x in range(GRID_SIZE):
            for y in range(GRID_SIZE): pygame.draw.rect(surface, GRID_COLOR, (x * cell_size, y * cell_size, cell_size, cell_size), 1)

        # Draw body
        for (x, y) in self.snake[1:]:
            pygame.draw.rect(surface, SNAKE_BODY_COLOR, (x * cell_size, y * cell_size, cell_size, cell_size))

        # Draw head
        if self.snake:
            head_pos = self.snake[0]
            # Angles for rotation: Up=0, Right=-90, Down=180, Left=90
            rotation_angles = {0: 0, 1: -90, 2: 180, 3: 90}
            angle = rotation_angles.get(self.direction, 0)
            rotated_head = pygame.transform.rotate(head_surface, angle)
            
            # Center the rotated head in the correct grid cell
            head_rect = rotated_head.get_rect()
            head_rect.center = (head_pos[0] * cell_size + cell_size / 2, head_pos[1] * cell_size + cell_size / 2)
            surface.blit(rotated_head, head_rect)

        pygame.draw.rect(surface, FOOD_COLOR, (self.food[0] * cell_size, self.food[1] * cell_size, cell_size, cell_size))

# --------------------- AI Core (No Changes)---------------------
class NeuralNetwork(nn.Module):
    def __init__(self, input_size, output_size):
        super().__init__(); self.net = nn.Sequential(nn.Linear(input_size, 16), nn.ReLU(), nn.Linear(16, output_size))
    def forward(self, x): return self.net(x)
class Agent:
    def __init__(self):
        self.network = NeuralNetwork(AI_INPUT_SIZE, 3); self.fitness = 0
    def select_action(self, state):
        with torch.no_grad(): return torch.argmax(self.network(state)).item()
class GeneticAlgorithm:
    def __init__(self):
        self.population = [Agent() for _ in range(POPULATION_SIZE)]
    def evolve_population(self, champion_state_dict=None):
        self.population.sort(key=lambda agent: agent.fitness, reverse=True)
        next_generation = []
        if champion_state_dict:
            champion = Agent(); champion.network.load_state_dict(champion_state_dict); next_generation.append(champion)
        elites_to_add = ELITISM_COUNT - len(next_generation)
        for i in range(elites_to_add): next_generation.append(self.population[i])
        while len(next_generation) < POPULATION_SIZE:
            parent1 = self._select_parent(); parent2 = self._select_parent()
            child = self._crossover(parent1, parent2); self._mutate(child)
            next_generation.append(child)
        self.population = next_generation
    def _select_parent(self):
        competitors = random.sample(self.population, 5)
        competitors.sort(key=lambda agent: agent.fitness, reverse=True); return competitors[0]
    def _crossover(self, parent1, parent2):
        child = Agent(); child_dict = child.network.state_dict()
        p1_dict, p2_dict = parent1.network.state_dict(), parent2.network.state_dict()
        for key in child_dict.keys():
            mask = torch.rand_like(child_dict[key]) > 0.5
            child_dict[key] = torch.where(mask, p1_dict[key], p2_dict[key])
        child.network.load_state_dict(child_dict); return child
    def _mutate(self, agent):
        with torch.no_grad():
            for param in agent.network.parameters():
                mask = torch.rand_like(param) < MUTATION_RATE
                mutation = torch.randn_like(param) * MUTATION_STRENGTH
                param.add_(mask * mutation)

# --------------------- Stats Tracker (No Changes)---------------------
class StatsTracker:
    def __init__(self):
        self.generation = 0; self.all_time_best_score = 0; self.avg_scores = []; self.max_scores = []; self.champion_network_state = None
    def update(self, population, games):
        self.generation += 1; scores = [game.score for game in games]; current_max_score = max(scores)
        if current_max_score > self.all_time_best_score:
            self.all_time_best_score = current_max_score
            champion_index = scores.index(current_max_score)
            self.champion_network_state = population[champion_index].network.state_dict().copy()
        self.max_scores.append(current_max_score); self.avg_scores.append(sum(scores) / len(scores))
    def get_surface(self, width, height, font):
        surface = pygame.Surface((width, height)); surface.fill(GRAPH_BACKGROUND)
        if len(self.avg_scores) < 2:
            msg = font.render("Awaiting data for graph...", True, TEXT_COLOR); surface.blit(msg, (width//2 - msg.get_width()//2, height//2 - msg.get_height()//2)); return surface
        title_font = pygame.font.Font(None, 28)
        title_surf = title_font.render("Score History by Generation", True, TEXT_COLOR); surface.blit(title_surf, (width//2 - title_surf.get_width()//2, 10))
        legend_y = title_surf.get_height() + 15
        pygame.draw.line(surface, GRAPH_MAX_COLOR, (width - 120, legend_y + 8), (width - 100, legend_y + 8), 2); surface.blit(font.render("Max Score", True, GRAPH_MAX_COLOR), (width-90, legend_y))
        pygame.draw.line(surface, GRAPH_LINE_COLOR, (width - 120, legend_y + 28), (width - 100, legend_y + 28), 2); surface.blit(font.render("Avg Score", True, GRAPH_LINE_COLOR), (width-90, legend_y + 20))
        plot_area = pygame.Rect(50, 70, width - 70, height - 110)
        all_scores = self.avg_scores + self.max_scores; max_val = max(all_scores) if all_scores else 1
        def plot_line(data, color):
            pts = []
            num_points = len(data)
            [pts.append((
                plot_area.x + int(i / (num_points - 1) * plot_area.width) if num_points > 1 else plot_area.x,
                plot_area.bottom - int(val / max_val * plot_area.height)
            )) for i, val in enumerate(data)]
            if len(pts) > 1:
                pygame.draw.lines(surface, color, False, pts, 2)
            elif len(pts) == 1:
                pygame.draw.circle(surface, color, pts[0], 3)
        plot_line(self.avg_scores, GRAPH_LINE_COLOR); plot_line(self.max_scores, GRAPH_MAX_COLOR)
        pygame.draw.line(surface, TEXT_COLOR, (plot_area.left, plot_area.bottom), (plot_area.left, plot_area.top), 1); pygame.draw.line(surface, TEXT_COLOR, (plot_area.left, plot_area.bottom), (plot_area.right, plot_area.bottom), 1)
        step_y = max(1, int(max_val / 5))
        for val in range(0, int(max_val) + 1, step_y):
            y = plot_area.bottom - int(val / max_val * plot_area.height); pygame.draw.line(surface, TEXT_COLOR, (plot_area.left - 5, y), (plot_area.left, y), 1)
            label = font.render(str(val), True, TEXT_COLOR); surface.blit(label, (plot_area.left - 30, y - label.get_height() // 2))
        num_generations = self.generation; step_x = max(1, (num_generations-1) // 5 if num_generations > 1 else 1)
        for i in range(1, num_generations + 1, step_x):
            x = plot_area.left + int((i - 1) / (num_generations - 1) * plot_area.width) if num_generations > 1 else plot_area.left
            pygame.draw.line(surface, TEXT_COLOR, (x, plot_area.bottom), (x, plot_area.bottom + 5), 1)
            label = font.render(str(i), True, TEXT_COLOR); surface.blit(label, (x - label.get_width() // 2, plot_area.bottom + 10))
        return surface

# --------------------- Main Simulation Loop ---------------------
def main():
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT)); pygame.display.set_caption("Genetic Algorithm Snake AI")
    clock = pygame.time.Clock()

    ga = GeneticAlgorithm(); games = [SnakeGame() for _ in range(POPULATION_SIZE)]; stats = StatsTracker()
    speed_multiplier = DEFAULT_SPEED; repetition_limit = DEFAULT_REPETITION_LIMIT
    paused = False; is_speed_slider_dragging = False; is_rep_slider_dragging = False
    speed_accumulator = 0.0

    cols, rows = 6, 6
    small_sim_width = LEFT_PANEL_WIDTH // cols; small_sim_height = WINDOW_HEIGHT // rows
    agent_surfaces = [pygame.Surface((small_sim_width, small_sim_height)) for _ in range(POPULATION_SIZE)]
    small_cell_size = min(small_sim_width, small_sim_height) // GRID_SIZE
    rendered_area_size = GRID_SIZE * small_cell_size

    # Create a dedicated, rotatable head surface (canonical form is facing up)
    head_surf = pygame.Surface((small_cell_size, small_cell_size), pygame.SRCALPHA)
    pygame.draw.rect(head_surf, SNAKE_HEAD_COLOR, (1, 1, small_cell_size - 2, small_cell_size - 2), border_radius=3)
    eye_r = max(1, small_cell_size // 8); eye_offset = small_cell_size // 4
    pygame.draw.circle(head_surf, EYE_COLOR, (eye_offset, eye_offset), eye_r)
    pygame.draw.circle(head_surf, EYE_COLOR, (small_cell_size - eye_offset, eye_offset), eye_r)
    
    dead_surf = pygame.Surface((rendered_area_size, rendered_area_size), pygame.SRCALPHA); dead_surf.fill(DEAD_COLOR)
    repetition_dead_surf = pygame.Surface((rendered_area_size, rendered_area_size), pygame.SRCALPHA)
    repetition_dead_surf.fill(REPETITION_DEAD_COLOR)

    small_font, med_font, large_font = pygame.font.Font(None, 22), pygame.font.Font(None, 28), pygame.font.Font(None, 36)

    running = True
    while running:
        control_area = pygame.Rect(LEFT_PANEL_WIDTH, 0, RIGHT_PANEL_WIDTH, CONTROL_AREA_HEIGHT)
        y_offset = control_area.y + 10
        y_offset += 50; y_offset += 25; y_offset += 20 * 4; y_offset += 25; y_offset += 25; y_offset += 40
        speed_slider_track = pygame.Rect(LEFT_PANEL_WIDTH + 20, y_offset, RIGHT_PANEL_WIDTH - 40, 14)
        speed_slider_hitbox = speed_slider_track.inflate(0, 20)
        y_offset += 65
        rep_slider_track = pygame.Rect(LEFT_PANEL_WIDTH + 20, y_offset, RIGHT_PANEL_WIDTH - 40, 14)
        rep_slider_hitbox = rep_slider_track.inflate(0, 20)

        mouse_pos = pygame.mouse.get_pos()
        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            elif event.type == pygame.KEYDOWN and event.key == pygame.K_SPACE: paused = not paused
            elif event.type == pygame.MOUSEBUTTONDOWN:
                if control_area.collidepoint(mouse_pos):
                    if speed_slider_hitbox.collidepoint(mouse_pos): is_speed_slider_dragging = True
                    if rep_slider_hitbox.collidepoint(mouse_pos): is_rep_slider_dragging = True
            elif event.type == pygame.MOUSEBUTTONUP:
                is_speed_slider_dragging = False; is_rep_slider_dragging = False
            elif event.type == pygame.MOUSEMOTION:
                if is_speed_slider_dragging:
                    handle_x = max(speed_slider_track.left, min(mouse_pos[0], speed_slider_track.right))
                    progress = (handle_x - speed_slider_track.left) / speed_slider_track.width
                    speed_multiplier = int(MIN_SPEED + progress * (MAX_SPEED - MIN_SPEED))
                if is_rep_slider_dragging:
                    handle_x = max(rep_slider_track.left, min(mouse_pos[0], rep_slider_track.right))
                    progress = (handle_x - rep_slider_track.left) / rep_slider_track.width
                    repetition_limit = int(MIN_REPETITION_LIMIT + progress * (MAX_REPETITION_LIMIT - MIN_REPETITION_LIMIT))
                    if repetition_limit % 2 != 0: repetition_limit += 1

        if not paused:
            speed_accumulator += speed_multiplier
            while speed_accumulator >= SPEED_DIVISOR:
                living_agents = False
                for i in range(POPULATION_SIZE):
                    if not games[i].is_done:
                        living_agents = True; state = games[i].get_state()
                        action = ga.population[i].select_action(state); games[i].step(action, repetition_limit)
                if not living_agents:
                    speed_accumulator = 0; break
                speed_accumulator -= SPEED_DIVISOR
            if all(game.is_done for game in games):
                for i in range(POPULATION_SIZE): ga.population[i].fitness = (2**games[i].score) + (games[i].total_steps / 500.0)
                stats.update(ga.population, games); ga.evolve_population(stats.champion_network_state)
                for game in games: game.reset()

        screen.fill(BACKGROUND_COLOR)
        leader_idx = max(range(POPULATION_SIZE), key=lambda i: games[i].score) if any(not g.is_done for g in games) else -1

        for i in range(POPULATION_SIZE):
            col, row = i % cols, i // rows; x_pos, y_pos = col * small_sim_width, row * small_sim_height
            game = games[i]

            game.render(agent_surfaces[i], small_cell_size, head_surf)
            screen.blit(agent_surfaces[i], (x_pos, y_pos))

            if game.is_done:
                if game.death_reason == 'repetition':
                    screen.blit(repetition_dead_surf, (x_pos, y_pos))
                else:
                    screen.blit(dead_surf, (x_pos, y_pos))

            score_surf = small_font.render(f"{game.score}", True, TEXT_COLOR); screen.blit(score_surf, (x_pos + 5, y_pos + 5))
            if not game.is_done and i == leader_idx:
                pygame.draw.rect(screen, LEADER_COLOR, (x_pos, y_pos, rendered_area_size, rendered_area_size), 2)

        stats_area = pygame.Rect(LEFT_PANEL_WIDTH, CONTROL_AREA_HEIGHT, RIGHT_PANEL_WIDTH, STATS_AREA_HEIGHT)
        pygame.draw.rect(screen, GRAPH_BACKGROUND, control_area);
        def draw_text(text, pos, font, color=TEXT_COLOR): screen.blit(font.render(text, True, color), pos)

        y_offset = control_area.y + 10
        status_text = "PAUSED" if paused else "SIMULATING"; status_color = (200, 50, 50) if paused else (50, 200, 50)
        draw_text(status_text, (control_area.centerx - large_font.size(status_text)[0]//2, y_offset), large_font, status_color); y_offset += 50

        last_gen_avg = f"{stats.avg_scores[-1]:.2f}" if stats.avg_scores else "N/A"
        last_gen_max = f"{stats.max_scores[-1]}" if stats.max_scores else "N/A"
        sections = {"Simulation Stats": [f"Generation: {stats.generation}", f"All-Time Best Score: {stats.all_time_best_score}", f"Last Gen Avg Score: {last_gen_avg}", f"Last Gen Max Score: {last_gen_max}"]}
        for title, lines in sections.items():
            draw_text(title, (control_area.x + 10, y_offset), med_font); y_offset += 25
            for line in lines: draw_text(line, (control_area.x + 15, y_offset), small_font); y_offset += 20
        y_offset += 25

        draw_text("Controls", (control_area.x + 10, y_offset), med_font); y_offset += 25
        draw_text("SPACE: Pause/Resume", (control_area.x + 15, y_offset), small_font); y_offset += 40

        slider_handle_rect = pygame.Rect(0, 0, 10, 24)
        progress = (speed_multiplier - MIN_SPEED) / (MAX_SPEED - MIN_SPEED) if (MAX_SPEED - MIN_SPEED) > 0 else 0
        slider_handle_rect.centerx = speed_slider_track.left + progress * speed_slider_track.width
        slider_handle_rect.centery = speed_slider_track.centery
        draw_text(f"Speed: x{speed_multiplier}", (speed_slider_track.left, speed_slider_track.top - 20), small_font)
        pygame.draw.rect(screen, SLIDER_TRACK_COLOR, speed_slider_track, border_radius=7)
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, slider_handle_rect, border_radius=3)

        slider_handle_rect = pygame.Rect(0, 0, 10, 24)
        progress = (repetition_limit - MIN_REPETITION_LIMIT) / (MAX_REPETITION_LIMIT - MIN_REPETITION_LIMIT) if (MAX_REPETITION_LIMIT - MIN_REPETITION_LIMIT) > 0 else 0
        slider_handle_rect.centerx = rep_slider_track.left + progress * rep_slider_track.width
        slider_handle_rect.centery = rep_slider_track.centery
        draw_text(f"Repetition Limit: {repetition_limit} moves", (rep_slider_track.left, rep_slider_track.top - 20), small_font)
        pygame.draw.rect(screen, SLIDER_TRACK_COLOR, rep_slider_track, border_radius=7)
        pygame.draw.rect(screen, SLIDER_HANDLE_COLOR, slider_handle_rect, border_radius=3)

        stats_surf = stats.get_surface(stats_area.width, stats_area.height, small_font); screen.blit(stats_surf, stats_area.topleft)
        pygame.draw.rect(screen, TEXT_COLOR, stats_area, 1)

        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()