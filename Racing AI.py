import pygame
import math
import numpy as np
import random
import sys
from collections import deque

# Global Parameters
NUM_CARS = 70
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
CAR_SPEED = 5  # Initial speed in units per frame
NUM_RAYS = 12  # Increased from 8 to 12 for better sensing
VISION_CONE_ANGLE = math.pi * 1.25  # Increased from pi/2 to pi*1.25 (225 degrees)
MIN_SPEED = 3
MAX_SPEED = 8
GENERATION_DURATION = 15 * 1000  # Increased from 10 to 15 seconds
CAR_WIDTH = 20
CAR_HEIGHT = 10
SIM_WIDTH = 850
SIM_HEIGHT = 600
NN_PANEL_WIDTH = 300
NN_PANEL_HEIGHT = 300
WIDTH = SIM_WIDTH + NN_PANEL_WIDTH
HEIGHT = SIM_HEIGHT
NN_PANEL_RECT = (SIM_WIDTH, (HEIGHT - NN_PANEL_HEIGHT) // 2, NN_PANEL_WIDTH, NN_PANEL_HEIGHT)

# Enhanced Physics Parameters
WHEELBASE = 5
MAX_STEERING_ANGLE = 0.2
ACCELERATION = 0.05
FRICTION = 0.02  # New parameter for simulating friction
CRASH_PENALTY = 10  # New parameter for penalizing crashes

# Pygame Initialization
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Neural Network Racing Simulation")
clock = pygame.time.Clock()
FONT_SMALL = pygame.font.Font(None, 24)
FONT_MEDIUM = pygame.font.Font(None, 36)

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

# Track Generation Functions
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

def generate_figure_eight_track(center, r, num_points, track_width):
    t = np.linspace(0, 2 * math.pi, num_points, endpoint=False)
    cx, cy = center
    centerline_points = [(cx + r * math.sin(ti), cy + r * math.sin(2 * ti)) for ti in t]
    left_points = []
    right_points = []
    for i in range(num_points):
        p0 = centerline_points[i]
        p1 = centerline_points[(i + 1) % num_points]
        dx = p1[0] - p0[0]
        dy = p1[1] - p0[1]
        magnitude = math.hypot(dx, dy)
        if magnitude > 1e-6:
            tangent = (dx / magnitude, dy / magnitude)
            normal = (-tangent[1], tangent[0])
            left_points.append((p0[0] + (track_width / 2) * normal[0], p0[1] + (track_width / 2) * normal[1]))
            right_points.append((p0[0] - (track_width / 2) * normal[0], p0[1] - (track_width / 2) * normal[1]))
        else:
            left_points.append(p0)
            right_points.append(p0)
    walls = []
    road_polygons = []
    for i in range(num_points):
        j = (i + 1) % num_points
        walls.append([left_points[i], left_points[j]])
        walls.append([right_points[i], right_points[j]])
        road_polygons.append([left_points[i], left_points[j], right_points[j], right_points[i]])
    checkpoints = []
    for i in range(0, num_points, 5):  # Increased checkpoint density (from 10 to 5)
        checkpoints.append([left_points[i], right_points[i]])
    start_idx = 25
    start_pos = centerline_points[start_idx]
    p0 = centerline_points[start_idx]
    p1 = centerline_points[(start_idx + 1) % num_points]
    dx = p1[0] - p0[0]
    dy = p1[1] - p0[1]
    start_angle = math.atan2(dy, dx)
    return walls, road_polygons, checkpoints, start_pos, start_angle

def set_track(track_type):
    global center, walls, checkpoints, road_polygons, start_pos, start_angle
    center = (SIM_WIDTH // 2, SIM_HEIGHT // 2)
    if track_type == "easy":
        outer_radius = 250
        inner_radius = 150
        num_segments = 40
        outer_walls = generate_circle_segments(center, outer_radius, num_segments)
        inner_walls = generate_circle_segments(center, inner_radius, num_segments)
        walls = outer_walls + inner_walls
        outer_points = [wall[0] for wall in outer_walls]
        inner_points = [wall[0] for wall in inner_walls]
        track_points = outer_points + inner_points[::-1]
        road_polygons = [track_points]
        checkpoints.clear()
        for i in range(20):  # Increased from 10 to 20 checkpoints
            angle = 2 * math.pi * i / 20
            p_inner = (center[0] + inner_radius * math.cos(angle), center[1] + inner_radius * math.sin(angle))
            p_outer = (center[0] + outer_radius * math.cos(angle), center[1] + outer_radius * math.sin(angle))
            checkpoints.append([p_inner, p_outer])
        start_pos = (center[0] + (inner_radius + outer_radius) / 2, center[1])
        start_angle = math.pi / 2
    elif track_type == "medium":
        outer_a = 250
        outer_b = 200
        inner_a = 150
        inner_b = 100
        num_segments = 40
        outer_walls = generate_ellipse_segments(center, outer_a, outer_b, num_segments)
        inner_walls = generate_ellipse_segments(center, inner_a, inner_b, num_segments)
        walls = outer_walls + inner_walls
        outer_points = [wall[0] for wall in outer_walls]
        inner_points = [wall[0] for wall in inner_walls]
        track_points = outer_points + inner_points[::-1]
        road_polygons = [track_points]
        checkpoints.clear()
        for i in range(20):  # Increased from 10 to 20 checkpoints
            angle = 2 * math.pi * i / 20
            p_inner = (center[0] + inner_a * math.cos(angle), center[1] + inner_b * math.sin(angle))
            p_outer = (center[0] + outer_a * math.cos(angle), center[1] + outer_b * math.sin(angle))
            checkpoints.append([p_inner, p_outer])
        start_pos = (center[0] + (inner_a + outer_a) / 2, center[1])
        start_angle = math.pi / 2
    elif track_type == "hard":
        r = 150
        num_points = 100
        track_width = 50
        walls, road_polygons, checkpoints, start_pos, start_angle = generate_figure_eight_track(
            center, r, num_points, track_width
        )

# Menu System
def menu():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1:
                    return "easy"
                elif event.key == pygame.K_2:
                    return "medium"
                elif event.key == pygame.K_3:
                    return "hard"
        screen.fill(WHITE)
        title_text = FONT_MEDIUM.render("Select Track Difficulty", True, BLACK)
        screen.blit(title_text, (SIM_WIDTH / 2 - title_text.get_width() / 2, 100))
        option1 = FONT_SMALL.render("1 - Easy (Circle Track)", True, BLACK)
        option2 = FONT_SMALL.render("2 - Medium (Oval Track)", True, BLACK)
        option3 = FONT_SMALL.render("3 - Hard (Figure-Eight Track)", True, BLACK)
        screen.blit(option1, (SIM_WIDTH / 2 - option1.get_width() / 2, 200))
        screen.blit(option2, (SIM_WIDTH / 2 - option2.get_width() / 2, 240))
        screen.blit(option3, (SIM_WIDTH / 2 - option3.get_width() / 2, 280))
        pygame.display.flip()
        clock.tick(60)

# Enhanced Neural Network Class
class NeuralNetwork:
    def __init__(self):
        # Input size: ray distances + speed + current steering angle + progress + relative checkpoint angle
        # [NUM_RAYS + 4] inputs in total
        self.input_size = NUM_RAYS + 4
        self.hidden_size = 24  # Increased from 16 to 24
        self.output_size = 6  # Still 6 outputs (3 steering, 3 acceleration)
        
        # Initialize weights with He initialization for better training
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0 / self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0 / self.hidden_size)
        self.b2 = np.zeros(self.output_size)
    
    def forward(self, x):
        # ReLU activation for hidden layer
        self.h = np.maximum(0, np.dot(x, self.w1) + self.b1)
        # Softmax for output layer
        o = np.dot(self.h, self.w2) + self.b2
        exp_o = np.exp(o - np.max(o))
        self.probs = exp_o / np.sum(exp_o)
        return self.probs
    
    def mutate(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        new_nn = self.clone()
        # Apply mutations with decaying strength based on network performance
        new_nn.w1 += np.random.randn(*self.w1.shape) * mutation_strength * (np.random.rand(*self.w1.shape) < mutation_rate)
        new_nn.b1 += np.random.randn(*self.b1.shape) * mutation_strength * (np.random.rand(*self.b1.shape) < mutation_rate)
        new_nn.w2 += np.random.randn(*self.w2.shape) * mutation_strength * (np.random.rand(*self.w2.shape) < mutation_rate)
        new_nn.b2 += np.random.randn(*self.b2.shape) * mutation_strength * (np.random.rand(*self.b2.shape) < mutation_rate)
        return new_nn

    def crossover(self, other):
        """Implement crossover between two neural networks"""
        child = NeuralNetwork()
        # Randomly choose weights from either parent
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

# Enhanced Car Class with Improved Physics
class Car:
    def __init__(self, nn):
        self.nn = nn
        self.x, self.y = start_pos
        self.angle = start_angle
        self.speed = CAR_SPEED
        self.steering_angle = 0
        self.acceleration = 0
        self.alive = True
        self.fitness = 0
        self.raw_fitness = 0  # Tracks fitness without penalties
        self.distance_traveled = 0
        self.checkpoint_times = {}  # Track when each checkpoint was reached
        self.current_checkpoint = 0
        self.prev_checkpoint = 0
        self.lap_count = 0
        self.prev_x = self.x
        self.prev_y = self.y
        self.idle_time = 0  # Track how long car has been inactive
        self.avg_speed = deque(maxlen=60)  # Track average speed
        self.wrong_way = False  # Flag for going the wrong way
        self.position_history = deque(maxlen=60)  # Track position history
        self.crashes = 0  # Count number of crashes
        self.start_time = pygame.time.get_ticks()
        
        # Initialize position history
        for _ in range(10):
            self.position_history.append((self.x, self.y))

    def calculate_angle_to_next_checkpoint(self):
        next_cp_idx = self.current_checkpoint
        cp = checkpoints[next_cp_idx]
        cp_x = (cp[0][0] + cp[1][0]) / 2
        cp_y = (cp[0][1] + cp[1][1]) / 2
        
        # Calculate angle to checkpoint
        dx = cp_x - self.x
        dy = cp_y - self.y
        cp_angle = math.atan2(dy, dx)
        
        # Calculate relative angle (-π to π)
        rel_angle = (cp_angle - self.angle) % (2 * math.pi)
        if rel_angle > math.pi:
            rel_angle -= 2 * math.pi
            
        return rel_angle / math.pi  # Normalize to range [-1, 1]

    def cast_rays(self):
        distances = []
        angle_start = self.angle - VISION_CONE_ANGLE / 2
        delta = VISION_CONE_ANGLE / (NUM_RAYS - 1)
        for i in range(NUM_RAYS):
            ray_angle = angle_start + i * delta
            dx = math.cos(ray_angle)
            dy = math.sin(ray_angle)
            min_t = float('inf')
            for wall in walls:
                ax, ay = wall[0]
                bx, by = wall[1]
                D = dx * (by - ay) - dy * (bx - ax)
                if abs(D) > 1e-6:
                    t = ((ay - self.y) * (bx - ax) - (ax - self.x) * (by - ay)) / D
                    s = (dx * (ay - self.y) - dy * (ax - self.x)) / D
                    if t >= 0 and 0 <= s <= 1:
                        min_t = min(min_t, t)
            dist = min(min_t, 200) if min_t < float('inf') else 200
            distances.append(dist / 200)  # Normalize distance
        return np.array(distances)

    def decide_actions(self):
        # Create enhanced input vector
        inputs = self.cast_rays()
        
        # Add additional inputs: speed, current steering angle, progress, and angle to next checkpoint
        normalized_speed = (self.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)  # Normalize to [0,1]
        normalized_steering = self.steering_angle / MAX_STEERING_ANGLE  # Normalize to [-1,1]
        checkpoint_progress = self.current_checkpoint / len(checkpoints)  # Progress through track
        checkpoint_angle = self.calculate_angle_to_next_checkpoint()
        
        # Combine all inputs
        input_vector = np.concatenate([
            inputs, 
            [normalized_speed, 
             normalized_steering, 
             checkpoint_progress, 
             checkpoint_angle]
        ])
        
        # Get NN outputs
        outputs = self.nn.forward(input_vector)
        steering_outputs = outputs[:3]  # Left, Straight, Right
        accel_outputs = outputs[3:]     # Decelerate, Maintain, Accelerate
        steering_action = np.argmax(steering_outputs)
        accel_action = np.argmax(accel_outputs)

        # Set steering angle based on neural network output
        if steering_action == 0:    # Left
            self.steering_angle = -MAX_STEERING_ANGLE
        elif steering_action == 1:  # Straight
            self.steering_angle = 0
        elif steering_action == 2:  # Right
            self.steering_angle = MAX_STEERING_ANGLE

        # Set acceleration based on neural network output
        if accel_action == 0:       # Decelerate
            self.acceleration = -ACCELERATION
        elif accel_action == 1:     # Maintain
            self.acceleration = 0
        elif accel_action == 2:     # Accelerate
            self.acceleration = ACCELERATION

    def update(self):
        if self.alive:
            # Save previous position
            self.prev_x = self.x
            self.prev_y = self.y
            
            # Update speed with acceleration and friction
            self.speed += self.acceleration
            self.speed -= FRICTION * self.speed  # Apply friction based on current speed
            self.speed = max(MIN_SPEED, min(MAX_SPEED, self.speed))
            
            # Add current speed to rolling average
            self.avg_speed.append(self.speed)

            # Update angle using bicycle model
            dtheta = (self.speed / WHEELBASE) * math.tan(self.steering_angle)
            self.angle += dtheta
            self.angle %= (2 * math.pi)

            # Update position
            self.x += self.speed * math.cos(self.angle)
            self.y += self.speed * math.sin(self.angle)
            
            # Track position history for detection of stalling
            self.position_history.append((self.x, self.y))
            
            # Calculate distance traveled
            self.distance_traveled += math.hypot(self.x - self.prev_x, self.y - self.prev_y)

            # Check for collisions and checkpoints
            if self.check_collision():
                self.alive = False
                self.crashes += 1
                self.fitness -= CRASH_PENALTY  # Apply crash penalty
            
            # Update checkpoint progress
            self.prev_checkpoint = self.current_checkpoint
            if self.check_checkpoint():
                new_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
                
                # Check if going backwards
                if new_checkpoint != (self.current_checkpoint + 1) % len(checkpoints):
                    self.wrong_way = True
                    self.fitness -= 5  # Penalty for going wrong way
                else:
                    self.wrong_way = False
                    self.current_checkpoint = new_checkpoint
                    current_time = pygame.time.get_ticks() - self.start_time
                    self.checkpoint_times[self.current_checkpoint] = current_time
                    
                    # Reward for reaching a new checkpoint
                    self.fitness += 10
                    self.raw_fitness += 1
                    
                    # Check for lap completion
                    if self.current_checkpoint == 0 and self.prev_checkpoint == len(checkpoints) - 1:
                        self.lap_count += 1
                        self.fitness += 50  # Bonus for completing a lap
            
            # Check for stalling/lack of progress
            if len(self.position_history) >= 60:
                oldest_pos = self.position_history[0]
                current_pos = self.position_history[-1]
                distance_moved = math.hypot(current_pos[0] - oldest_pos[0], current_pos[1] - oldest_pos[1])
                
                if distance_moved < 30:  # If car hasn't moved much in the last 60 frames
                    self.idle_time += 1
                    if self.idle_time > 120:  # If car has been idle for 2 seconds (120 frames)
                        self.alive = False
                        self.fitness -= 20  # Penalty for stalling
                else:
                    self.idle_time = 0
            
            # Add small continuous reward for speed (encourages going fast)
            self.fitness += 0.001 * self.speed
            
            # Add small continuous reward for staying in the middle of the track
            # (This would require calculating distance to track edges, omitted for brevity)

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
        p1, q1 = seg1
        p2, q2 = seg2
        o1 = self.orientation(p1, q1, p2)
        o2 = self.orientation(p1, q1, q2)
        o3 = self.orientation(p2, q2, p1)
        o4 = self.orientation(p2, q2, q1)
        return o1 != o2 and o3 != o4

    def orientation(self, p, q, r):
        val = (q[1] - p[1]) * (r[0] - q[0]) - (q[0] - p[0]) * (r[1] - q[1])
        if abs(val) < 1e-6:
            return 0
        return 1 if val > 0 else 2

# Drawing Functions
def draw_track():
    for poly in road_polygons:
        pygame.draw.polygon(screen, GREY, poly)
    for wall in walls:
        pygame.draw.line(screen, BLACK, wall[0], wall[1], 2)
    for cp in checkpoints:
        pygame.draw.line(screen, GREEN, cp[0], cp[1], 1)

def draw_neural_network(panel_rect, nn, inputs):
    if not hasattr(nn, 'h') or not hasattr(nn, 'probs'):
        # If forward hasn't been called yet, run a forward pass
        nn.forward(inputs)
    
    h = nn.h
    probs = nn.probs
    
    panel_x, panel_y, panel_w, panel_h = panel_rect
    input_x = panel_x + 50
    hidden_x = panel_x + panel_w // 2
    output_x = panel_x + panel_w - 50
    
    n_input = nn.input_size
    n_hidden = nn.hidden_size
    n_output = nn.output_size
    
    spacing_input = panel_h / (n_input + 1)
    spacing_hidden = panel_h / (n_hidden + 1)
    spacing_output = panel_h / (n_output + 1)
    
    input_neurons = [(input_x, panel_y + (i + 1) * spacing_input) for i in range(n_input)]
    hidden_neurons = [(hidden_x, panel_y + (i + 1) * spacing_hidden) for i in range(n_hidden)]
    output_neurons = [(output_x, panel_y + (i + 1) * spacing_output) for i in range(n_output)]
    
    # Draw weights connecting input to hidden layer
    for i, inp_pos in enumerate(input_neurons):
        for j, hid_pos in enumerate(hidden_neurons):
            weight = nn.w1[i, j]
            color = BLUE if weight > 0 else RED
            width = max(1, int(abs(weight) * 2))
            # Only draw strong connections to reduce visual clutter
            if abs(weight) > 0.3:
                pygame.draw.line(screen, color, inp_pos, hid_pos, width)
    
    # Draw weights connecting hidden to output layer
    for i, hid_pos in enumerate(hidden_neurons):
        for j, out_pos in enumerate(output_neurons):
            weight = nn.w2[i, j]
            color = BLUE if weight > 0 else RED
            width = max(1, int(abs(weight) * 2))
            # Only draw strong connections
            if abs(weight) > 0.3:
                pygame.draw.line(screen, color, hid_pos, out_pos, width)
    
    # Draw input neurons
    for i, pos in enumerate(input_neurons):
        if i < NUM_RAYS:
            # Color based on ray distance
            activation = inputs[i]
            color_val = int(activation * 255)
            pygame.draw.circle(screen, (color_val, color_val, 0), pos, 6)
        else:
            # Different color for metadata inputs
            activation = inputs[i]
            color_val = int(abs(activation) * 255)
            pygame.draw.circle(screen, (0, color_val, color_val), pos, 6)
        
        pygame.draw.circle(screen, BLACK, pos, 6, 1)
    
    # Draw hidden neurons
    for i, pos in enumerate(hidden_neurons):
        activation = h[i]
        color_val = int(min(activation, 1) * 255)
        pygame.draw.circle(screen, (0, color_val, 0), pos, 6)
        pygame.draw.circle(screen, BLACK, pos, 6, 1)
    
    # Draw output neurons with labels
    output_labels = ["Left", "Straight", "Right", "Slow", "Maintain", "Fast"]
    for i, pos in enumerate(output_neurons):
        activation = probs[i]
        color_val = int(activation * 255)
        pygame.draw.circle(screen, (0, 0, color_val), pos, 8)
        pygame.draw.circle(screen, BLACK, pos, 8, 1)
        
        # Add output labels
        label = output_labels[i]
        text = FONT_SMALL.render(label, True, BLACK)
        if i < 3:  # Steering actions
            screen.blit(text, (pos[0] + 10, pos[1] - 8))
        else:      # Speed actions
            screen.blit(text, (pos[0] + 10, pos[1] - 8))

def draw_car(car):
    color = RED
    if not car.alive:
        color = (100, 100, 100)  # Grey for dead cars
    
    # Draw the car body
    w, h = CAR_WIDTH, CAR_HEIGHT
    local_corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    rotated = []
    for (x, y) in local_corners:
        rx = x * math.cos(car.angle) - y * math.sin(car.angle)
        ry = x * math.sin(car.angle) + y * math.cos(car.angle)
        rotated.append((car.x + rx, car.y + ry))
    pygame.draw.polygon(screen, color, rotated)
    pygame.draw.polygon(screen, BLACK, rotated, 1)
    
    # Draw front direction indicator
    front_x = car.x + (w/2) * math.cos(car.angle)
    front_y = car.y + (w/2) * math.sin(car.angle)
    pygame.draw.circle(screen, BLACK, (int(front_x), int(front_y)), 3)

# Enhanced Selection Methods
def tournament_selection(cars, k=3):
    """Tournament selection: randomly select k cars and return the best one"""
    selected = []
    for _ in range(k):
        idx = random.randint(0, len(cars) - 1)
        selected.append(cars[idx])
    return max(selected, key=lambda car: car.fitness)

def roulette_wheel_selection(cars):
    """Roulette wheel selection based on fitness"""
    total_fitness = sum(max(0.1, car.fitness) for car in cars)  # Avoid negative fitness
    if total_fitness <= 0:
        return random.choice(cars)
    pick = random.uniform(0, total_fitness)
    current = 0
    for car in cars:
        current += max(0.1, car.fitness)
        if current >= pick:
            return car
    return cars[-1]  # Fallback

def select_parents(cars, selection_method="tournament"):
    """Select parents for reproduction using the specified method"""
    if selection_method == "tournament":
        return tournament_selection(cars)
    else:
        return roulette_wheel_selection(cars)

# Elite Selection - keep the best performing cars
def elitism(cars, count=5):
    """Select the top performing cars to keep unchanged"""
    sorted_cars = sorted(cars, key=lambda x: x.fitness, reverse=True)
    return [car.nn.clone() for car in sorted_cars[:count]]

# Main Simulation Loop
def run_simulation():
    global walls, checkpoints, road_polygons, start_pos, start_angle

    # Select track from menu
    track_type = menu()
    set_track(track_type)

    # Initialize first generation of cars with random NNs
    cars = [Car(NeuralNetwork()) for _ in range(NUM_CARS)]
    generation = 1
    best_fitness = -float('inf')
    best_nn = None
    
    # Generation timer
    generation_start_time = pygame.time.get_ticks()
    
    # Statistics tracking
    generation_stats = []
    
    running = True
    paused = False
    show_all_cars = True  # Toggle to only show best car
    
    # Main game loop
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_SPACE:
                    paused = not paused
                elif event.key == pygame.K_a:
                    show_all_cars = not show_all_cars
                elif event.key == pygame.K_n:
                    # Skip to next generation
                    generation_start_time = 0
        
        if paused:
            pygame.time.delay(100)
            continue

        # Clear screen
        screen.fill(WHITE)
        
        # Draw track
        draw_track()
        
        # Update and draw cars
        alive_count = 0
        best_car = None
        best_fitness_current = -float('inf')
        
        for car in cars:
            if car.alive:
                # Get actions from neural network
                car.decide_actions()
                # Update car state
                car.update()
                alive_count += 1
                
                # Track best car in current generation
                if car.fitness > best_fitness_current:
                    best_fitness_current = car.fitness
                    best_car = car
        
        # Draw cars (either all or just the best)
        if show_all_cars:
            for car in cars:
                draw_car(car)
        elif best_car:
            draw_car(best_car)
        
        # Update all-time best NN if we have a new champion
        if best_fitness_current > best_fitness:
            best_fitness = best_fitness_current
            if best_car:
                best_nn = best_car.nn.clone()
        
        # Draw neural network visualization for best car
        if best_car:
            # Prepare inputs for visualization
            ray_distances = best_car.cast_rays()
            normalized_speed = (best_car.speed - MIN_SPEED) / (MAX_SPEED - MIN_SPEED)
            normalized_steering = best_car.steering_angle / MAX_STEERING_ANGLE
            checkpoint_progress = best_car.current_checkpoint / len(checkpoints)
            checkpoint_angle = best_car.calculate_angle_to_next_checkpoint()
            
            inputs = np.concatenate([
                ray_distances, 
                [normalized_speed, normalized_steering, checkpoint_progress, checkpoint_angle]
            ])
            
            draw_neural_network(NN_PANEL_RECT, best_car.nn, inputs)
        
        # Display stats
        stats = [
            f"Generation: {generation}",
            f"Cars Alive: {alive_count}/{NUM_CARS}",
            f"Best Fitness: {best_fitness:.2f}",
            f"Current Best: {best_fitness_current:.2f}",
            f"Time: {(pygame.time.get_ticks() - generation_start_time) / 1000:.1f}s"
        ]
        
        if best_car:
            stats.extend([
                f"Speed: {best_car.speed:.2f}",
                f"Lap Count: {best_car.lap_count}",
                f"Checkpoint: {best_car.current_checkpoint}/{len(checkpoints)}",
                f"Distance: {best_car.distance_traveled:.1f}"
            ])
        
        # Draw stats text
        for i, text in enumerate(stats):
            text_surface = FONT_SMALL.render(text, True, BLACK)
            screen.blit(text_surface, (10, 10 + i * 25))
        
        # Check if generation time is up or all cars are dead
        current_time = pygame.time.get_ticks()
        generation_elapsed = current_time - generation_start_time
        
        if generation_elapsed > GENERATION_DURATION or alive_count == 0:
            # Record stats for this generation
            avg_fitness = sum(car.fitness for car in cars) / NUM_CARS
            generation_stats.append({
                'generation': generation,
                'best_fitness': best_fitness_current,
                'avg_fitness': avg_fitness,
                'alive_count': alive_count
            })
            
            # Create next generation
            new_generation = []
            
            # Elitism: keep the best performers
            elite_count = max(3, NUM_CARS // 10)  # 10% of population or at least 3
            elite_nns = elitism(cars, elite_count)
            
            for nn in elite_nns:
                new_generation.append(Car(nn))
            
            # Fill rest with offspring from selection, crossover, and mutation
            while len(new_generation) < NUM_CARS:
                # Tournament or roulette wheel selection
                parent1 = select_parents(cars, "tournament")
                parent2 = select_parents(cars, "tournament")
                
                # Crossover
                if random.random() < 0.7:  # 70% chance of crossover
                    child_nn = parent1.nn.crossover(parent2.nn)
                else:
                    # No crossover, just clone one parent
                    child_nn = parent1.nn.clone() if random.random() < 0.5 else parent2.nn.clone()
                
                # Mutation (with adaptive rate based on generation)
                mutation_rate = MUTATION_RATE * (1.0 - min(0.5, generation / 100))
                child_nn = child_nn.mutate(mutation_rate=mutation_rate)
                
                # Add to new generation
                new_generation.append(Car(child_nn))
            
            # Replace old generation with new
            cars = new_generation
            generation += 1
            
            # Reset generation timer
            generation_start_time = pygame.time.get_ticks()
            
            # Display generation change message
            gen_msg = f"Generation {generation} started!"
            gen_text = FONT_MEDIUM.render(gen_msg, True, BLUE)
            screen.blit(gen_text, (SIM_WIDTH // 2 - gen_text.get_width() // 2, 50))
            pygame.display.flip()
            pygame.time.delay(1000)  # Pause briefly to show generation change
        
        pygame.display.flip()
        clock.tick(60)

    pygame.quit()

if __name__ == "__main__":
    run_simulation()