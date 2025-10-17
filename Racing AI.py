import pygame
import math
import numpy as np
import random
import sys
from collections import deque
import os # <<< NEW FEATURE: To check if saved file exists >>>

# Global Parameters
NUM_CARS = 100
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.45

NUM_RAYS = 8  # Reduced from 12 to 8 rays
VISION_CONE_ANGLE = math.pi * 0.8  # Narrower: 144 degrees instead of 225

CONSTANT_SPEED = 320  # Set a single constant speed for all cars

# Corrected Timings: GENERATION_DURATION is now a simple 20 seconds.
GENERATION_DURATION = 20

# NOTE: GENERATION_FRAMES is no longer used for timing the simulation.

CAR_WIDTH = 22
CAR_HEIGHT = 11

# Simulation dimensions
SIM_WIDTH = 850
SIM_HEIGHT = 600
HEIGHT = SIM_HEIGHT

UI_WIDTH = 320 # Give a bit more width on the right side
WIDTH = SIM_WIDTH + UI_WIDTH  # Overall window width

# NN Panel
NN_PANEL_WIDTH = 280
NN_PANEL_HEIGHT = 300
NN_PANEL_X = SIM_WIDTH + (UI_WIDTH - NN_PANEL_WIDTH) // 2
NN_PANEL_Y = 50
NN_PANEL_RECT = (NN_PANEL_X, NN_PANEL_Y, NN_PANEL_WIDTH, NN_PANEL_HEIGHT)

# Graph
GRAPH_WIDTH = 310
GRAPH_HEIGHT = 215
GRAPH_X = SIM_WIDTH - 15 + (UI_WIDTH - GRAPH_WIDTH) // 2
GRAPH_Y = NN_PANEL_Y + NN_PANEL_HEIGHT + 20
GRAPH_RECT = (GRAPH_X, GRAPH_Y, GRAPH_WIDTH, GRAPH_HEIGHT)

BRIGHT_GREEN = (0, 255, 0)  # Highlight for alive leading car
DARK_GREEN = (0, 100, 0)    # Highlight for dead leading car

# Enhanced Physics Parameters
WHEELBASE = 5
MAX_STEERING_ANGLE = 0.21  # 12 degrees in radians
CRASH_PENALTY = 10  # New parameter for penalizing crashes

# Pygame Initialization
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Neural Network Racing Simulation")
clock = pygame.time.Clock()
FONT_SMALL = pygame.font.Font(None, 24)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_LARGE = pygame.font.Font(None, 48) # <<< NEW FEATURE: For main menu title >>>

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
GREEN = (0, 255, 0)
BLUE = (0, 0, 255)
GREY = (200, 200, 200)
YELLOW = (255, 255, 0)
PURPLE = (128, 0, 128)

# Track Globals
center = (SIM_WIDTH // 2, SIM_HEIGHT // 2)
walls = []
checkpoints = []
road_polygons = []
start_pos = None
start_angle = None

# Track Generation Functions (Unchanged)
def generate_circle_segments(center, radius, num_segments):
    segments = []
    cx, cy = center
    for i in range(num_segments):
        theta1 = 2 * math.pi * i / num_segments
        theta2 = 2 * math.pi * (i + 1) / num_segments
        p1 = (cx + radius * math.cos(theta1), cy + radius * math.sin(theta1))
        p2 = (cx + radius * math.cos(theta2), cy + radius * math.sin(theta2))
        segments.append([p1, p2])
    return segments

def generate_ellipse_segments(center, a, b, num_segments):
    segments = []
    cx, cy = center
    for i in range(num_segments):
        t1 = 2 * math.pi * i / num_segments
        t2 = 2 * math.pi * (i + 1) / num_segments
        p1 = (cx + a * math.cos(t1), cy + b * math.sin(t1))
        p2 = (cx + a * math.cos(t2), cy + b * math.sin(t2))
        segments.append([p1, p2])
    return segments

def generate_square_track(center, size, track_width):
    cx, cy = center
    length_multiplier, skinnier_factor = 1.5, 0.95
    adjusted_track_width = track_width * skinnier_factor
    outer_width, outer_height = size * length_multiplier, size
    inner_width, inner_height = outer_width - adjusted_track_width, outer_height - adjusted_track_width
    outer_corners = [(cx - outer_width, cy - outer_height), (cx + outer_width, cy - outer_height), (cx + outer_width, cy + outer_height), (cx - outer_width, cy + outer_height)]
    inner_corners = [(cx - inner_width, cy - inner_height), (cx + inner_width, cy - inner_height), (cx + inner_width, cy + inner_height), (cx - inner_width, cy + inner_height)]
    points_per_short_edge, points_per_long_edge = 20, int(20 * length_multiplier)
    outer_points, inner_points = [], []
    for j in range(points_per_long_edge): t = j / points_per_long_edge; outer_points.append((outer_corners[0][0] + t * (outer_corners[1][0] - outer_corners[0][0]), outer_corners[0][1]))
    for j in range(points_per_short_edge): t = j / points_per_short_edge; outer_points.append((outer_corners[1][0], outer_corners[1][1] + t * (outer_corners[2][1] - outer_corners[1][1])))
    for j in range(points_per_long_edge): t = j / points_per_long_edge; outer_points.append((outer_corners[2][0] - t * (outer_corners[2][0] - outer_corners[3][0]), outer_corners[2][1]))
    for j in range(points_per_short_edge): t = j / points_per_short_edge; outer_points.append((outer_corners[3][0], outer_corners[3][1] - t * (outer_corners[3][1] - outer_corners[0][1])))
    for j in range(points_per_long_edge): t = j / points_per_long_edge; inner_points.append((inner_corners[0][0] + t * (inner_corners[1][0] - inner_corners[0][0]), inner_corners[0][1]))
    for j in range(points_per_short_edge): t = j / points_per_short_edge; inner_points.append((inner_corners[1][0], inner_corners[1][1] + t * (inner_corners[2][1] - inner_corners[1][1])))
    for j in range(points_per_long_edge): t = j / points_per_long_edge; inner_points.append((inner_corners[2][0] - t * (inner_corners[2][0] - inner_corners[3][0]), inner_corners[2][1]))
    for j in range(points_per_short_edge): t = j / points_per_short_edge; inner_points.append((inner_corners[3][0], inner_corners[3][1] - t * (inner_corners[3][1] - inner_corners[0][1])))
    centerline_points = [((op[0] + ip[0]) / 2, (op[1] + ip[1]) / 2) for op, ip in zip(outer_points, inner_points)]
    walls = []
    for i in range(len(outer_points)): walls.append([outer_points[i], outer_points[(i + 1) % len(outer_points)]])
    for i in range(len(inner_points)): walls.append([inner_points[i], inner_points[(i + 1) % len(inner_points)]])
    road_polygons = [outer_points + inner_points[::-1]]
    checkpoints = [[inner_points[i], outer_points[i]] for i in range(0, len(outer_points), len(outer_points) // 25)]
    start_idx = 0
    start_pos = centerline_points[start_idx]
    p0, p1 = centerline_points[start_idx], centerline_points[(start_idx + 1) % len(centerline_points)]
    start_angle = math.atan2(p1[1] - p0[1], p1[0] - p0[0])
    return walls, road_polygons, checkpoints, start_pos, start_angle

def set_track(track_type):
    global center, walls, checkpoints, road_polygons, start_pos, start_angle
    if track_type == "easy":
        center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2)
        outer_radius, inner_radius, num_segments = 250, 150, 40
        outer_walls = generate_circle_segments(center, outer_radius, num_segments)
        inner_walls = generate_circle_segments(center, inner_radius, num_segments)
        walls = outer_walls + inner_walls
        outer_points = [w[0] for w in outer_walls]; inner_points = [w[0] for w in inner_walls]
        road_polygons = [outer_points + inner_points[::-1]]
        checkpoints.clear()
        for i in range(20):
            angle = 2 * math.pi * i / 20
            p_inner = (center[0] + inner_radius * math.cos(angle), center[1] + inner_radius * math.sin(angle))
            p_outer = (center[0] + outer_radius * math.cos(angle), center[1] + outer_radius * math.sin(angle))
            checkpoints.append([p_inner, p_outer])
        start_pos = (center[0] + (inner_radius + outer_radius) / 2, center[1])
        start_angle = math.pi / 2
    elif track_type == "medium":
        center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2)
        outer_a, outer_b, inner_a, inner_b, num_segments = 250, 200, 170, 120, 40
        outer_walls = generate_ellipse_segments(center, outer_a, outer_b, num_segments)
        inner_walls = generate_ellipse_segments(center, inner_a, inner_b, num_segments)
        walls = outer_walls + inner_walls
        outer_points = [w[0] for w in outer_walls]; inner_points = [w[0] for w in inner_walls]
        road_polygons = [outer_points + inner_points[::-1]]
        checkpoints.clear()
        for i in range(20):
            angle = 2 * math.pi * i / 20
            p_inner = (center[0] + inner_a * math.cos(angle), center[1] + inner_b * math.sin(angle))
            p_outer = (center[0] + outer_a * math.cos(angle), center[1] + outer_b * math.sin(angle))
            checkpoints.append([p_inner, p_outer])
        start_pos = (center[0] + (inner_a + outer_a) / 2, center[1])
        start_angle = math.pi / 2
    elif track_type == "hard":
        hard_center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2 + 50)
        square_size, track_width = 175, 70
        walls, road_polygons, checkpoints, start_pos, start_angle = generate_square_track(hard_center, square_size, track_width)

# Menu System
def track_select_menu(): # <<< NEW FEATURE: Renamed for clarity >>>
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: return "easy"
                elif event.key == pygame.K_2: return "medium"
                elif event.key == pygame.K_3: return "hard"
                elif event.key == pygame.K_4: return "impossible"
        screen.fill(WHITE)
        title_text = FONT_MEDIUM.render("Select Track Difficulty", True, BLACK)
        screen.blit(title_text, (WIDTH / 2 - title_text.get_width() / 2, 100))
        option1 = FONT_SMALL.render("1 - Easy (Circle Track)", True, BLACK)
        option2 = FONT_SMALL.render("2 - Medium (Oval Track)", True, BLACK)
        option3 = FONT_SMALL.render("3 - Hard (Square Track)", True, BLACK)
        option4 = FONT_SMALL.render("4 - Train on Random Tracks", True, BLACK)
        screen.blit(option1, (WIDTH / 2 - option1.get_width() / 2, 200))
        screen.blit(option2, (WIDTH / 2 - option2.get_width() / 2, 240))
        screen.blit(option3, (WIDTH / 2 - option3.get_width() / 2, 280))
        screen.blit(option4, (WIDTH / 2 - option4.get_width() / 2, 320))
        pygame.display.flip()
        clock.tick(60)

# <<< NEW FEATURE: Main menu to choose between training and testing >>>
def main_menu():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "TRAIN"
                elif event.key == pygame.K_2:
                    if os.path.exists("best_car_nn.npz"):
                        return "TEST"
                    else:
                        print("No saved AI found! Please train an AI first.")
        
        screen.fill(WHITE)
        title_text = FONT_LARGE.render("Racing AI Simulation", True, BLACK)
        screen.blit(title_text, (WIDTH / 2 - title_text.get_width() / 2, 100))

        option1 = FONT_MEDIUM.render("1 - Train New AI", True, BLACK)
        screen.blit(option1, (WIDTH / 2 - option1.get_width() / 2, 250))

        # Check if a saved file exists to enable the test option
        if os.path.exists("best_car_nn.npz"):
            option2 = FONT_MEDIUM.render("2 - Test Saved AI", True, BLACK)
        else:
            option2 = FONT_MEDIUM.render("2 - Test Saved AI (No file found)", True, GREY)
        
        screen.blit(option2, (WIDTH / 2 - option2.get_width() / 2, 310))

        pygame.display.flip()
        clock.tick(60)

# Enhanced Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.input_size = 8 + 3
        self.hidden_size = 14
        self.output_size = 3
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.output_size)

    # <<< NEW FEATURE: Save and Load methods >>>
    def save(self, filename="best_car_nn.npz"):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        print(f"Neural network saved to {filename}")

    @staticmethod
    def load(filename="best_car_nn.npz"):
        data = np.load(filename)
        nn = NeuralNetwork()
        nn.w1 = data['w1']
        nn.b1 = data['b1']
        nn.w2 = data['w2']
        nn.b2 = data['b2']
        print(f"Neural network loaded from {filename}")
        return nn

    def forward(self, x):
        self.h = np.maximum(0, np.dot(x, self.w1) + self.b1)
        o = np.dot(self.h, self.w2) + self.b2
        exp_o = np.exp(o - np.max(o))
        self.probs = exp_o / np.sum(exp_o)
        return self.probs

    def mutate(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        new_nn = self.clone()
        new_nn.w1 += np.random.randn(*self.w1.shape) * mutation_strength * (np.random.rand(*self.w1.shape) < mutation_rate)
        new_nn.b1 += np.random.randn(*self.b1.shape) * mutation_strength * (np.random.rand(*self.b1.shape) < mutation_rate)
        new_nn.w2 += np.random.randn(*self.w2.shape) * mutation_strength * (np.random.rand(*self.w2.shape) < mutation_rate)
        new_nn.b2 += np.random.randn(*self.b2.shape) * mutation_strength * (np.random.rand(*self.b2.shape) < mutation_rate)
        return new_nn

    def crossover(self, other):
        child = NeuralNetwork()
        mask1 = np.random.rand(*self.w1.shape) < 0.5
        mask2 = np.random.rand(*self.w2.shape) < 0.5
        child.w1 = np.where(mask1, self.w1, other.w1)
        child.b1 = np.where(np.random.rand(*self.b1.shape) < 0.5, self.b1, other.b1)
        child.w2 = np.where(mask2, self.w2, other.w2)
        child.b2 = np.where(np.random.rand(*self.b2.shape) < 0.5, self.b2, other.b2)
        return child

    def clone(self):
        clone_nn = NeuralNetwork()
        clone_nn.w1 = np.copy(self.w1)
        clone_nn.b1 = np.copy(self.b1)
        clone_nn.w2 = np.copy(self.w2)
        clone_nn.b2 = np.copy(self.b2)
        return clone_nn

# Car Class (Unchanged)
class Car:
    def __init__(self, nn):
        self.last_checkpoint_time = pygame.time.get_ticks()
        self.nn = nn
        self.x, self.y = start_pos
        self.angle = start_angle
        self.speed = CONSTANT_SPEED
        self.steering_angle = 0
        self.alive = True
        self.fitness = 0
        self.raw_fitness = 0
        self.distance_traveled = 0
        self.checkpoint_times = {}
        self.current_checkpoint = 0
        self.prev_checkpoint = 0
        self.lap_count = 0
        self.prev_x = self.x
        self.prev_y = self.y
        self.idle_time = 0
        self.avg_speed = deque(maxlen=60)
        self.wrong_way = False
        self.position_history = deque(maxlen=60)
        self.crashes = 0
        self.start_time = pygame.time.get_ticks()
        for _ in range(10): self.position_history.append((self.x, self.y))

    def calculate_angle_to_next_checkpoint(self):
        next_cp_idx = self.current_checkpoint
        cp = checkpoints[next_cp_idx]
        cp_x = (cp[0][0] + cp[1][0]) / 2; cp_y = (cp[0][1] + cp[1][1]) / 2
        dx = cp_x - self.x; dy = cp_y - self.y
        cp_angle = math.atan2(dy, dx)
        rel_angle = (cp_angle - self.angle) % (2 * math.pi)
        if rel_angle > math.pi: rel_angle -= 2 * math.pi
        return rel_angle / math.pi

    def cast_rays(self):
        distances = []
        angle_start = self.angle - VISION_CONE_ANGLE / 2
        delta = VISION_CONE_ANGLE / (NUM_RAYS - 1)
        for i in range(NUM_RAYS):
            ray_angle = angle_start + i * delta
            dx = math.cos(ray_angle); dy = math.sin(ray_angle)
            min_t = float('inf')
            for wall in walls:
                ax, ay = wall[0]; bx, by = wall[1]
                D = dx * (by - ay) - dy * (bx - ax)
                if abs(D) > 1e-6:
                    t = ((ay - self.y) * (bx - ax) - (ax - self.x) * (by - ay)) / D
                    s = (dx * (ay - self.y) - dy * (ax - self.x)) / D
                    if t >= 0 and 0 <= s <= 1: min_t = min(min_t, t)
            dist = min(min_t, 200) if min_t < float('inf') else 200
            distances.append(dist / 200)
        return np.array(distances)

    def decide_actions(self):
        inputs = self.cast_rays()
        normalized_steering = self.steering_angle / MAX_STEERING_ANGLE
        checkpoint_progress = self.current_checkpoint / len(checkpoints)
        checkpoint_angle = self.calculate_angle_to_next_checkpoint()
        input_vector = np.concatenate([inputs, [normalized_steering, checkpoint_progress, checkpoint_angle]])
        outputs = self.nn.forward(input_vector)
        steering_action = np.argmax(outputs)
        if steering_action == 0: self.steering_angle = -MAX_STEERING_ANGLE
        elif steering_action == 1: self.steering_angle = 0
        elif steering_action == 2: self.steering_angle = MAX_STEERING_ANGLE

    def update(self, delta_time):
        if self.alive:
            self.prev_x = self.x; self.prev_y = self.y
            self.speed = CONSTANT_SPEED
            self.avg_speed.append(self.speed)
            dtheta = (self.speed / WHEELBASE) * math.tan(self.steering_angle) * delta_time
            self.angle += dtheta; self.angle %= (2 * math.pi)
            self.x += self.speed * math.cos(self.angle) * delta_time
            self.y += self.speed * math.sin(self.angle) * delta_time
            self.position_history.append((self.x, self.y))
            self.distance_traveled += math.hypot(self.x - self.prev_x, self.y - self.prev_y)
            if self.check_collision():
                self.alive = False; self.crashes += 1; self.fitness -= CRASH_PENALTY
            self.prev_checkpoint = self.current_checkpoint
            if self.check_checkpoint():
                new_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
                if new_checkpoint != (self.current_checkpoint + 1) % len(checkpoints):
                    self.wrong_way = True; self.fitness -= 5
                else:
                    self.wrong_way = False
                    self.current_checkpoint = new_checkpoint
                    self.last_checkpoint_time = pygame.time.get_ticks()
                    self.fitness += 10; self.raw_fitness += 1
                    if self.current_checkpoint == 0 and self.prev_checkpoint == len(checkpoints) - 1:
                        self.lap_count += 1; self.fitness += 50
            current_time = pygame.time.get_ticks()
            if current_time - self.last_checkpoint_time > 5000 and self.alive:
                self.alive = False; self.fitness -= 10
            if len(self.position_history) >= 60:
                oldest_pos = self.position_history[0]; current_pos = self.position_history[-1]
                distance_moved = math.hypot(current_pos[0] - oldest_pos[0], current_pos[1] - oldest_pos[1])
                if distance_moved < 30:
                    self.idle_time += 1
                    if self.idle_time > 120: self.alive = False; self.fitness -= 20
                else: self.idle_time = 0

    def check_collision(self):
        car_path = [(self.prev_x, self.prev_y), (self.x, self.y)]
        for wall in walls:
            if self.intersects(car_path, wall):
                return True
        return False
    def check_checkpoint(self):
        checkpoint = checkpoints[self.current_checkpoint]
        return self.intersects([(self.prev_x, self.prev_y), (self.x, self.y)], checkpoint)
    def intersects(self, seg1, seg2):
        p1, q1 = seg1; p2, q2 = seg2
        o1 = self.orientation(p1, q1, p2); o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1); o4 = self.orientation(p2, q2, q1)
        return o1 != o2 and o3 != o4
    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-6: return 0
        return 1 if val > 0 else 2

# Drawing Functions (Unchanged)
def draw_track():
    for poly in road_polygons: pygame.draw.polygon(screen, GREY, poly)
    for wall in walls: pygame.draw.line(screen, BLACK, wall[0], wall[1], 2)
    for cp in checkpoints: pygame.draw.line(screen, GREEN, cp[0], cp[1], 1)
def draw_performance_graph(rect, stats):
    x, y, w, h = rect
    pygame.draw.rect(screen, WHITE, rect); pygame.draw.rect(screen, BLACK, rect, 1)
    title = FONT_SMALL.render("Performance History", True, BLACK)
    screen.blit(title, (x + (w - title.get_width()) // 2, y + 5))
    if len(stats) < 2:
        msg = FONT_SMALL.render("Collecting data...", True, (100, 100, 100))
        screen.blit(msg, (x + (w - msg.get_width()) // 2, y + h // 2))
        return
    y_min = 0.0; generations = [s['generation'] for s in stats]; best_fitness, avg_fitness = [], []
    for s in stats:
        bf = s['best_fitness']; best_fitness.append(bf if math.isfinite(bf) else y_min)
        af = s['avg_fitness']; avg_fitness.append(af if math.isfinite(af) else y_min)
    valid_best = [f for f in best_fitness if math.isfinite(f)]; valid_avg = [f for f in avg_fitness if math.isfinite(f)]
    y_max = max(max(valid_best), max(valid_avg)) * 1.1 if valid_best and valid_avg else 1.0
    if y_max <= y_min: y_max = y_min + 1.0
    graph_x, graph_y, graph_w, graph_h = x + 40, y + 25, w - 50, h - 60
    pygame.draw.line(screen, BLACK, (graph_x, graph_y), (graph_x, graph_y + graph_h), 1)
    pygame.draw.line(screen, BLACK, (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1)
    for i in range(5):
        fraction = i / 4.0; value = y_min + fraction * (y_max - y_min)
        y_pos = graph_y + graph_h - int(fraction * graph_h)
        label = FONT_SMALL.render(f"{value:.0f}", True, BLACK)
        screen.blit(label, (graph_x - label.get_width() - 5, y_pos - label.get_height() // 2))
    total_gens, max_gen = len(generations), generations[-1] if generations else 0
    for i in range(min(total_gens, 5)):
        idx = i * (total_gens - 1) // 4 if total_gens > 1 else 0; gen_label = generations[idx]
        denominator = max_gen - generations[0] if generations else 1; denominator = 1 if denominator == 0 else denominator
        fraction = (gen_label - generations[0]) / denominator; x_pos = graph_x + int(fraction * graph_w)
        label = FONT_SMALL.render(f"{gen_label}", True, BLACK)
        screen.blit(label, (x_pos - label.get_width() // 2, graph_y + graph_h + 5))
    def fitness_to_y(f):
        if y_max - y_min < 1e-9 or not math.isfinite(f): return graph_y + graph_h // 2
        return graph_y + graph_h - int(((f - y_min) / (y_max - y_min)) * graph_h)
    best_points = [(graph_x + int(((gen - generations[0]) / (1 if (max_gen - generations[0])==0 else (max_gen - generations[0]))) * graph_w), fitness_to_y(best_fitness[i])) for i, gen in enumerate(generations)]
    if len(best_points) > 1: pygame.draw.lines(screen, (0, 0, 255), False, best_points, 2)
    avg_points = [(graph_x + int(((gen - generations[0]) / (1 if (max_gen - generations[0])==0 else (max_gen - generations[0]))) * graph_w), fitness_to_y(avg_fitness[i])) for i, gen in enumerate(generations)]
    if len(avg_points) > 1: pygame.draw.lines(screen, (0, 200, 0), False, avg_points, 2)
    legend_x, legend_y, line_length = graph_x - 140, graph_y + 150, 20
    pygame.draw.line(screen, (0, 0, 255), (legend_x, legend_y), (legend_x + line_length, legend_y), 2)
    best_label = FONT_SMALL.render("Best", True, (0, 0, 255)); screen.blit(best_label, (legend_x + line_length + 8, legend_y - 8))
    legend_y += 25
    pygame.draw.line(screen, (0, 200, 0), (legend_x, legend_y), (legend_x + line_length, legend_y), 2)
    avg_label = FONT_SMALL.render("Average", True, (0, 200, 0)); screen.blit(avg_label, (legend_x + line_length + 8, legend_y - 8))
def draw_neural_network(panel_rect, nn, inputs):
    if not hasattr(nn, 'h') or not hasattr(nn, 'probs'): nn.forward(inputs)
    h, probs = nn.h, nn.probs
    panel_x, panel_y, panel_w, panel_h = panel_rect
    input_x, hidden_x, output_x = panel_x + 40, panel_x + panel_w // 2, panel_x + panel_w - 40
    n_input, n_hidden, n_output = nn.input_size, nn.hidden_size, nn.output_size
    spacing_input, spacing_hidden, spacing_output = panel_h / (n_input + 1), panel_h / (n_hidden + 1), panel_h / (n_output + 1)
    input_neurons = [(input_x, panel_y + (i + 1) * spacing_input) for i in range(n_input)]
    hidden_neurons = [(hidden_x, panel_y + (i + 1) * spacing_hidden) for i in range(n_hidden)]
    output_neurons = [(output_x, panel_y + (i + 1) * spacing_output) for i in range(n_output)]
    pygame.draw.rect(screen, WHITE, panel_rect); pygame.draw.rect(screen, BLACK, panel_rect, 1)
    title = FONT_SMALL.render("Neural Network", True, BLACK)
    screen.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y - 20))
    for i, inp_pos in enumerate(input_neurons):
        for j, hid_pos in enumerate(hidden_neurons):
            weight = nn.w1[i, j]
            if abs(weight) > 0.4: color = BLUE if weight > 0 else RED; width = max(1, int(abs(weight) * 2)); pygame.draw.line(screen, color, inp_pos, hid_pos, width)
    for i, hid_pos in enumerate(hidden_neurons):
        for j, out_pos in enumerate(output_neurons):
            weight = nn.w2[i, j]
            if abs(weight) > 0.4: color = BLUE if weight > 0 else RED; width = max(1, int(abs(weight) * 2)); pygame.draw.line(screen, color, hid_pos, out_pos, width)
    node_size = 4
    for i, pos in enumerate(input_neurons):
        if i < NUM_RAYS: activation = inputs[i]; color_val = int(activation * 255); pygame.draw.circle(screen, (color_val, color_val, 0), pos, node_size)
        else: activation = inputs[i]; color_val = int(abs(activation) * 255); pygame.draw.circle(screen, (0, color_val, color_val), pos, node_size)
        pygame.draw.circle(screen, BLACK, pos, node_size, 1)
    for i, pos in enumerate(hidden_neurons): activation = h[i]; color_val = int(min(activation, 1) * 255); pygame.draw.circle(screen, (0, color_val, 0), pos, node_size); pygame.draw.circle(screen, BLACK, pos, node_size, 1)
    output_labels = ["L", "S", "R"]
    for i, pos in enumerate(output_neurons):
        activation = probs[i]; color_val = int(activation * 255); pygame.draw.circle(screen, (0, 0, color_val), pos, node_size + 2); pygame.draw.circle(screen, BLACK, pos, node_size + 2, 1)
        label = output_labels[i]; text = FONT_SMALL.render(label, True, BLACK); screen.blit(text, (pos[0] + 8, pos[1] - 8))
def draw_car(car, is_best=False):
    if is_best: color = BRIGHT_GREEN if car.alive else DARK_GREEN
    else: color = RED if car.alive else (100, 100, 100)
    w, h = CAR_WIDTH, CAR_HEIGHT
    local_corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    rotated = [(car.x + (x * math.cos(car.angle) - y * math.sin(car.angle)), car.y + (x * math.sin(car.angle) + y * math.cos(car.angle))) for x,y in local_corners]
    pygame.draw.polygon(screen, color, rotated)
    pygame.draw.polygon(screen, BLACK, rotated, 1)
    front_x = car.x + (w/2) * math.cos(car.angle); front_y = car.y + (w/2) * math.sin(car.angle)
    pygame.draw.circle(screen, BLACK, (int(front_x), int(front_y)), 3)
# Enhanced Selection Methods (Unchanged)
def tournament_selection(cars, k=5): return max(random.sample(cars, k), key=lambda car: car.fitness)
def roulette_wheel_selection(cars):
    total_fitness = sum(max(0.1, car.fitness) for car in cars)
    if total_fitness <= 0: return random.choice(cars)
    pick = random.uniform(0, total_fitness)
    current = 0
    for car in cars:
        current += max(0.1, car.fitness)
        if current >= pick: return car
    return cars[-1]
def select_parents(cars, selection_method="tournament"):
    if selection_method == "tournament": return tournament_selection(cars)
    else: return roulette_wheel_selection(cars)
def elitism(cars, count=5): return [car.nn.clone() for car in sorted(cars, key=lambda x: x.fitness, reverse=True)[:count]]

def run_simulation(mode="TRAIN"): # <<< NEW FEATURE: mode parameter >>>
    # This function is now the main application loop, managing menu and simulation states.
    # Part 1 is now handled by main_menu()
    
    # Part 2: SIMULATION SETUP
    if mode == "TRAIN":
        track_type = track_select_menu()
        if track_type == "impossible":
            track_options = ["easy", "medium", "hard"]
            current_track = random.choice(track_options)
            previous_track = current_track
        else:
            track_options = [track_type]
            current_track = track_type
            previous_track = None
        set_track(current_track)
        cars = [Car(NeuralNetwork()) for _ in range(NUM_CARS)]
    else: # mode == "TEST"
        track_type = track_select_menu()
        if track_type == "impossible": track_type = "easy" # Default to easy for testing
        set_track(track_type)
        saved_nn = NeuralNetwork.load()
        cars = [Car(saved_nn)] # Only one car

    is_first_frame = True
    generation = 1
    best_fitness = -float('inf')
    generation_stats = []
    paused = False
    show_all_cars = True
    save_notification_timer = 0 # <<< NEW FEATURE: Timer for save message >>>

    back_button_rect = pygame.Rect(10, SIM_HEIGHT - 40, 160, 30)
    back_button_color = (220, 220, 220); back_button_hover_color = (200, 200, 200)

    generation_start_time = pygame.time.get_ticks()

    # Part 3: SIMULATION LOOP
    simulation_running = True
    while simulation_running:
        delta_time = clock.tick(60) / 1000.0
        delta_time = min(delta_time, 0.05)
        if save_notification_timer > 0:
            save_notification_timer -= delta_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: simulation_running = False
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_a: show_all_cars = not show_all_cars
                # <<< NEW FEATURE: Save on 'S' key press in TRAIN mode >>>
                elif event.key == pygame.K_s and mode == "TRAIN":
                    best_car = max(cars, key=lambda car: car.fitness) if cars else None
                    if best_car:
                        best_car.nn.save()
                        save_notification_timer = 2 # Show message for 2 seconds
                elif event.key == pygame.K_n and mode == "TRAIN":
                    generation_start_time = 0
            if event.type == pygame.MOUSEBUTTONDOWN:
                if back_button_rect.collidepoint(event.pos): simulation_running = False

        if paused: pygame.time.delay(100); continue

        screen.fill(WHITE)
        draw_track()

        alive_count = 0
        if not is_first_frame:
            for car in cars:
                if car.alive:
                    car.decide_actions()
                    car.update(delta_time)
                    alive_count += 1
        else: alive_count = len(cars)

        best_car = max(cars, key=lambda car: car.fitness) if cars else None
        if mode == "TEST" and best_car and not best_car.alive:
             # In test mode, reset the car if it crashes
            saved_nn = best_car.nn 
            set_track(track_type) # Reset track to reset start pos
            cars = [Car(saved_nn)]
            best_car = cars[0]

        best_fitness_current = best_car.fitness if best_car else -float('inf')

        if show_all_cars:
            for car in cars: draw_car(car, is_best=(car == best_car))
        elif best_car:
            draw_car(best_car, is_best=True)

        if is_first_frame: is_first_frame = False

        if best_fitness_current > best_fitness: best_fitness = best_fitness_current

        if best_car:
            ray_distances = best_car.cast_rays()
            normalized_steering = best_car.steering_angle / MAX_STEERING_ANGLE
            checkpoint_progress = best_car.current_checkpoint / len(checkpoints)
            checkpoint_angle = best_car.calculate_angle_to_next_checkpoint()
            inputs = np.concatenate([ray_distances, [normalized_steering, checkpoint_progress, checkpoint_angle]])
            draw_neural_network(NN_PANEL_RECT, best_car.nn, inputs)
            if mode == "TRAIN":
                draw_performance_graph(GRAPH_RECT, generation_stats)

        mouse_pos = pygame.mouse.get_pos()
        if back_button_rect.collidepoint(mouse_pos):
            pygame.draw.rect(screen, back_button_hover_color, back_button_rect, border_radius=5)
        else: pygame.draw.rect(screen, back_button_color, back_button_rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, back_button_rect, 1, border_radius=5)
        back_text = FONT_SMALL.render("Back to Menu (Esc)", True, BLACK)
        text_rect = back_text.get_rect(center=back_button_rect.center)
        screen.blit(back_text, text_rect)

        elapsed_seconds = (pygame.time.get_ticks() - generation_start_time) / 1000.0

        stats = [f"FPS: {clock.get_fps():.1f}"]
        if mode == "TRAIN":
             stats.extend([
                f"Generation: {generation}",
                f"Time: {elapsed_seconds:.2f}s / {GENERATION_DURATION:.0f}s",
                f"Cars Alive: {alive_count}/{NUM_CARS}",
                f"Best Fitness: {best_fitness:.2f}",
                f"Current Best: {best_fitness_current:.2f}",
                f"Press 'S' to save best car"
             ])
        else: # Test mode stats
            stats.append("--- TEST MODE ---")

        if best_car:
            stats.extend([
                f"Speed: {best_car.speed:.2f}",
                f"Lap Count: {best_car.lap_count}",
                f"Checkpoint: {best_car.current_checkpoint}/{len(checkpoints)}",
                f"Distance: {best_car.distance_traveled:.1f}"
            ])
        if track_type == "impossible" and mode == "TRAIN":
            stats.append(f"Current Track: {current_track}")

        for i, text in enumerate(stats):
            text_surface = FONT_SMALL.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 24))
        
        # <<< NEW FEATURE: Show save notification >>>
        if save_notification_timer > 0:
            save_text = FONT_MEDIUM.render("Best AI Saved!", True, BLUE)
            screen.blit(save_text, (SIM_WIDTH / 2 - save_text.get_width() / 2, 10))


        # Generation End Condition: Only for TRAIN mode
        if mode == "TRAIN" and (elapsed_seconds >= GENERATION_DURATION or alive_count == 0):
            best_fitness_current = max(car.fitness for car in cars) if cars else -float('inf')
            avg_fitness = sum(car.fitness for car in cars) / NUM_CARS if cars else 0
            generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness_current,
                'avg_fitness': avg_fitness,
                'alive_count': alive_count
            })

            if track_type == "impossible":
                available_tracks = [t for t in track_options if t != previous_track]
                new_track = random.choice(available_tracks) if available_tracks else random.choice(track_options)
                current_track, previous_track = new_track, new_track
                set_track(current_track)
            else:
                set_track(current_track)

            new_generation = []
            elite_count = max(3, NUM_CARS // 10)
            elite_nns = elitism(cars, elite_count)
            for nn in elite_nns: new_generation.append(Car(nn))

            top_cars = sorted(cars, key=lambda x: x.fitness, reverse=True)[:5]
            for i, top_car in enumerate(top_cars):
                num_mutants = 2
                if i == 0: num_mutants += 2
                for _ in range(num_mutants):
                    mutant_nn = top_car.nn.mutate()
                    new_generation.append(Car(mutant_nn))

            while len(new_generation) < NUM_CARS:
                parent1, parent2 = select_parents(cars), select_parents(cars)
                child_nn = parent1.nn.crossover(parent2.nn) if random.random() < 0.85 else (parent1.nn.clone() if random.random() < 0.5 else parent2.nn.clone())
                mutation_rate = MUTATION_RATE * (1.0 - min(0.5, generation / 100))
                child_nn = child_nn.mutate(mutation_rate=mutation_rate)
                new_generation.append(Car(child_nn))

            cars = new_generation
            generation += 1
            is_first_frame = True
            generation_start_time = pygame.time.get_ticks()

            gen_msg = f"Generation {generation} starting!"
            gen_text = FONT_MEDIUM.render(gen_msg, True, BLUE)
            screen.blit(gen_text, (SIM_WIDTH // 2 - gen_text.get_width() / 2, 50))
            pygame.display.flip()
            pygame.time.delay(1000)

        pygame.display.flip()

if __name__ == "__main__":
    while True:
        mode = main_menu()
        run_simulation(mode)