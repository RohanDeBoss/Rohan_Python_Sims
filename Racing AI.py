import pygame
import math
import numpy as np
import random
import sys
from collections import deque

# Global Parameters
NUM_CARS = 100
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.45

NUM_RAYS = 8  # Reduced from 12 to 8 rays
VISION_CONE_ANGLE = math.pi * 0.8  # Narrower: 144 degrees instead of 225

CONSTANT_SPEED = 320  # Set a single constant speed for all cars

GENERATION_DURATION = 20 / 2 # 20 Seconds

# With a frame-based duration
GENERATION_FRAMES = GENERATION_DURATION * 60  # 15 seconds at 60 fps

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

def generate_square_track(center, size, track_width):

    cx, cy = center
    
    # Make the track longer by applying a multiplier to the horizontal dimension
    length_multiplier = 1.5  # Adjust this to make the track longer
    
    # Make the track skinnier by reducing the track width
    skinnier_factor = 0.95  # Adjust this to make the track skinnier
    adjusted_track_width = track_width * skinnier_factor
    
    # Define the dimensions for outer boundary
    outer_width = size * length_multiplier
    outer_height = size
    
    # Define the dimensions for inner boundary
    inner_width = outer_width - adjusted_track_width
    inner_height = outer_height - adjusted_track_width
    
    # Define the exact points for the outer and inner rectangles
    # Outer rectangle corners (clockwise from top-left)
    outer_corners = [
        (cx - outer_width, cy - outer_height),  # top-left
        (cx + outer_width, cy - outer_height),  # top-right
        (cx + outer_width, cy + outer_height),  # bottom-right
        (cx - outer_width, cy + outer_height),  # bottom-left
    ]
    
    # Inner rectangle corners (clockwise from top-left)
    inner_corners = [
        (cx - inner_width, cy - inner_height),  # top-left
        (cx + inner_width, cy - inner_height),  # top-right
        (cx + inner_width, cy + inner_height),  # bottom-right
        (cx - inner_width, cy + inner_height),  # bottom-left
    ]
    
    # Number of points to generate along each edge for smoother rendering
    # Use more points for longer edges
    points_per_short_edge = 20
    points_per_long_edge = int(points_per_short_edge * length_multiplier)
    
    # Generate points along the outer rectangle
    outer_points = []
    # Top edge (longer)
    for j in range(points_per_long_edge):
        t = j / points_per_long_edge
        x = outer_corners[0][0] + t * (outer_corners[1][0] - outer_corners[0][0])
        y = outer_corners[0][1]
        outer_points.append((x, y))
    
    # Right edge (shorter)
    for j in range(points_per_short_edge):
        t = j / points_per_short_edge
        x = outer_corners[1][0]
        y = outer_corners[1][1] + t * (outer_corners[2][1] - outer_corners[1][1])
        outer_points.append((x, y))
    
    # Bottom edge (longer)
    for j in range(points_per_long_edge):
        t = j / points_per_long_edge
        x = outer_corners[2][0] - t * (outer_corners[2][0] - outer_corners[3][0])
        y = outer_corners[2][1]
        outer_points.append((x, y))
    
    # Left edge (shorter)
    for j in range(points_per_short_edge):
        t = j / points_per_short_edge
        x = outer_corners[3][0]
        y = outer_corners[3][1] - t * (outer_corners[3][1] - outer_corners[0][1])
        outer_points.append((x, y))
    
    # Generate points along the inner rectangle
    inner_points = []
    # Top edge (longer)
    for j in range(points_per_long_edge):
        t = j / points_per_long_edge
        x = inner_corners[0][0] + t * (inner_corners[1][0] - inner_corners[0][0])
        y = inner_corners[0][1]
        inner_points.append((x, y))
    
    # Right edge (shorter)
    for j in range(points_per_short_edge):
        t = j / points_per_short_edge
        x = inner_corners[1][0]
        y = inner_corners[1][1] + t * (inner_corners[2][1] - inner_corners[1][1])
        inner_points.append((x, y))
    
    # Bottom edge (longer)
    for j in range(points_per_long_edge):
        t = j / points_per_long_edge
        x = inner_corners[2][0] - t * (inner_corners[2][0] - inner_corners[3][0])
        y = inner_corners[2][1]
        inner_points.append((x, y))
    
    # Left edge (shorter)
    for j in range(points_per_short_edge):
        t = j / points_per_short_edge
        x = inner_corners[3][0]
        y = inner_corners[3][1] - t * (inner_corners[3][1] - inner_corners[0][1])
        inner_points.append((x, y))
    
    # Generate centerline points by averaging inner and outer points
    centerline_points = []
    for i in range(len(outer_points)):
        x = (outer_points[i][0] + inner_points[i][0]) / 2
        y = (outer_points[i][1] + inner_points[i][1]) / 2
        centerline_points.append((x, y))
    
    # Create walls for inner and outer boundaries
    walls = []
    
    # Create walls along outer boundary
    for i in range(len(outer_points)):
        j = (i + 1) % len(outer_points)
        walls.append([outer_points[i], outer_points[j]])
    
    # Create walls along inner boundary
    for i in range(len(inner_points)):
        j = (i + 1) % len(inner_points)
        walls.append([inner_points[i], inner_points[j]])
    
    # Create road polygon
    road_poly = []
    # Add outer points in order
    for p in outer_points:
        road_poly.append(p)
    # Add inner points in reverse order to create a continuous polygon
    for p in reversed(inner_points):
        road_poly.append(p)
    
    road_polygons = [road_poly]
    
    # Create checkpoints - perpendicular lines across track at regular intervals
    checkpoints = []
    total_points = len(outer_points)
    checkpoint_spacing = total_points // 25  # Changed from 16 to 25 for more frequent checkpoints

    for i in range(0, total_points, checkpoint_spacing):
        checkpoints.append([inner_points[i], outer_points[i]])

    # Set starting position and angle
    start_idx = 0  # Start at first checkpoint (top left corner)
    start_pos = centerline_points[start_idx]
    p0 = centerline_points[start_idx]
    p1 = centerline_points[(start_idx + 1) % len(centerline_points)]
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
        inner_a = 170
        inner_b = 120
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
        hard_center = (center[0] + 50, center[1] + 50)  # New center: (475, 350)
        square_size = 175
        track_width = 70
        walls, road_polygons, checkpoints, start_pos, start_angle = generate_square_track(
            hard_center, square_size, track_width
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
        option3 = FONT_SMALL.render("3 - Hard (Square Track)", True, BLACK)  # Updated description
        screen.blit(option1, (SIM_WIDTH / 2 - option1.get_width() / 2, 200))
        screen.blit(option2, (SIM_WIDTH / 2 - option2.get_width() / 2, 240))
        screen.blit(option3, (SIM_WIDTH / 2 - option3.get_width() / 2, 280))
        pygame.display.flip()
        clock.tick(60)

# Enhanced Neural Network Class
class NeuralNetwork:
    def __init__(self):
        # Reduce inputs: fewer rays (7) + steering + progress + checkpoint angle
        self.input_size = 8 + 3
        # Smaller hidden layer: approximately mean of input and output size
        self.hidden_size = 14
        self.output_size = 3  # Keep the same (left, straight, right)
        
        # Initialize weights with He initialization
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
        self.last_checkpoint_time = pygame.time.get_ticks()
        self.nn = nn
        self.x, self.y = start_pos
        self.angle = start_angle
        self.speed = CONSTANT_SPEED  # Use constant speed
        self.steering_angle = 0
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
        self.avg_speed = deque(maxlen=60)  # Keep for position tracking
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
        # Create simplified input vector
        inputs = self.cast_rays()
        
        # Add additional inputs: current steering angle, progress, and angle to next checkpoint
        # Removed normalized_speed
        normalized_steering = self.steering_angle / MAX_STEERING_ANGLE  # Normalize to [-1,1]
        checkpoint_progress = self.current_checkpoint / len(checkpoints)  # Progress through track
        checkpoint_angle = self.calculate_angle_to_next_checkpoint()
        
        # Combine all inputs
        input_vector = np.concatenate([
            inputs, 
            [normalized_steering, 
            checkpoint_progress, 
            checkpoint_angle]
        ])
        
        # Get NN outputs - only steering outputs
        outputs = self.nn.forward(input_vector)
        steering_outputs = outputs  # Left, Straight, Right
        steering_action = np.argmax(steering_outputs)

        # Set steering angle based on neural network output
        if steering_action == 0:    # Left
            self.steering_angle = -MAX_STEERING_ANGLE
        elif steering_action == 1:  # Straight
            self.steering_angle = 0
        elif steering_action == 2:  # Right
            self.steering_angle = MAX_STEERING_ANGLE

    def update(self, delta_time):
        if self.alive:
            # Save previous position
            self.prev_x = self.x
            self.prev_y = self.y
            
            # Speed is now constant
            self.speed = CONSTANT_SPEED
            
            # Add current position to rolling history
            self.avg_speed.append(self.speed)

            # Update angle using bicycle model (scaled by delta_time)
            dtheta = (self.speed / WHEELBASE) * math.tan(self.steering_angle) * delta_time
            self.angle += dtheta
            self.angle %= (2 * math.pi)

            # Update position (movement scaled by delta_time)
            self.x += self.speed * math.cos(self.angle) * delta_time
            self.y += self.speed * math.sin(self.angle) * delta_time

            # Track position history for stalling detection
            self.position_history.append((self.x, self.y))
            
            # Calculate distance traveled (automatically delta_time-aware)
            self.distance_traveled += math.hypot(
                self.x - self.prev_x,
                self.y - self.prev_y
            )

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
                    self.last_checkpoint_time = pygame.time.get_ticks()  # Reset timer
                    
                    # Reward for reaching a new checkpoint
                    self.fitness += 10
                    self.raw_fitness += 1
                    
                    # Check for lap completion
                    if self.current_checkpoint == 0 and self.prev_checkpoint == len(checkpoints) - 1:
                        self.lap_count += 1
                        self.fitness += 50  # Bonus for completing a lap
            
            # Check if car takes too long to reach the next checkpoint
            current_time = pygame.time.get_ticks()
            if current_time - self.last_checkpoint_time > 5000 and self.alive:
                self.alive = False
                self.fitness -= 10  # Optional penalty for timing out
            
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

def draw_performance_graph(rect, stats):
    x, y, w, h = rect
    
    # Draw background and border
    pygame.draw.rect(screen, WHITE, rect)
    pygame.draw.rect(screen, BLACK, rect, 1)
    
    # Title
    title = FONT_SMALL.render("Performance History", True, BLACK)
    screen.blit(title, (x + (w - title.get_width()) // 2, y + 5))
    
    if len(stats) < 2:
        msg = FONT_SMALL.render("Collecting data...", True, (100, 100, 100))
        screen.blit(msg, (x + (w - msg.get_width()) // 2, y + h // 2))
        return
    
    # Extract data with validation
    generations = [s['generation'] for s in stats]
    best_fitness = []
    avg_fitness = []
    for s in stats:
        bf = s['best_fitness']
        best_fitness.append(bf if math.isfinite(bf) else y_min)  # Fixed here
        af = s['avg_fitness']
        avg_fitness.append(af if math.isfinite(af) else y_min)    # Fixed here
    
    # Calculate y-axis bounds with safety checks
    valid_best = [f for f in best_fitness if math.isfinite(f)]
    valid_avg = [f for f in avg_fitness if math.isfinite(f)]
    if valid_best and valid_avg:
        max_best = max(valid_best)
        max_avg = max(valid_avg)
        y_max = max(max_best, max_avg) * 1.1
    else:
        y_max = 1.0  # Default value if all are invalid
    
    y_min = 0.0  # Explicitly define y_min
    if y_max <= y_min:
        y_max = y_min + 1.0
    
    graph_x = x + 40
    graph_y = y + 25
    graph_w = w - 50
    graph_h = h - 60
    
    # Draw axes
    pygame.draw.line(screen, BLACK, (graph_x, graph_y), (graph_x, graph_y + graph_h), 1)
    pygame.draw.line(screen, BLACK, (graph_x, graph_y + graph_h), (graph_x + graph_w, graph_y + graph_h), 1)

    # Y-axis labels
    for i in range(5):
        fraction = i / 4.0
        value = y_min + fraction * (y_max - y_min)
        y_pos = graph_y + graph_h - int(fraction * graph_h)
        
        label = FONT_SMALL.render(f"{value:.0f}", True, BLACK)
        screen.blit(label, (graph_x - label.get_width() - 5, y_pos - label.get_height() // 2))

    # X-axis labels
    total_gens = len(generations)
    max_gen = generations[-1] if generations else 0
    for i in range(min(total_gens, 5)):
        idx = i * (total_gens - 1) // 4 if total_gens > 1 else 0
        gen_label = generations[idx]
        denominator = max_gen - generations[0] if generations else 1
        denominator = denominator if denominator != 0 else 1
        fraction = (gen_label - generations[0]) / denominator
        x_pos = graph_x + int(fraction * graph_w)
        
        label = FONT_SMALL.render(f"{gen_label}", True, BLACK)
        screen.blit(label, (x_pos - label.get_width() // 2, graph_y + graph_h + 5))

    def fitness_to_y(f):
        if y_max - y_min < 1e-9 or not math.isfinite(f):
            return graph_y + graph_h // 2
        normalized = (f - y_min) / (y_max - y_min)
        return graph_y + graph_h - int(normalized * graph_h)

    # Plot best fitness
    best_points = []
    for i, gen in enumerate(generations):
        denominator = max_gen - generations[0] if generations else 1
        denominator = denominator if denominator != 0 else 1
        px = graph_x + int((gen - generations[0]) / denominator * graph_w)
        py = fitness_to_y(best_fitness[i])
        best_points.append((px, py))
    if len(best_points) > 1:
        pygame.draw.lines(screen, (0, 0, 255), False, best_points, 2)

    # Plot average fitness
    avg_points = []
    for i, gen in enumerate(generations):
        denominator = max_gen - generations[0] if generations else 1
        denominator = denominator if denominator != 0 else 1
        px = graph_x + int((gen - generations[0]) / denominator * graph_w)
        py = fitness_to_y(avg_fitness[i])
        avg_points.append((px, py))
    if len(avg_points) > 1:
        pygame.draw.lines(screen, (0, 200, 0), False, avg_points, 2)

    # Legend (moved further left)
    legend_x = graph_x - 140  # Shifted left by 60px from graph's left edge
    legend_y = graph_y + 150
    line_length = 20

    # Best fitness legend
    pygame.draw.line(screen, (0, 0, 255), 
                    (legend_x, legend_y), 
                    (legend_x + line_length, legend_y), 2)
    best_label = FONT_SMALL.render("Best", True, (0, 0, 255))
    screen.blit(best_label, (legend_x + line_length + 8, legend_y - 8))  # 8px padding

    # Average fitness legend
    legend_y += 25  # Vertical spacing between entries
    pygame.draw.line(screen, (0, 200, 0), 
                    (legend_x, legend_y), 
                    (legend_x + line_length, legend_y), 2)
    avg_label = FONT_SMALL.render("Average", True, (0, 200, 0))
    screen.blit(avg_label, (legend_x + line_length + 8, legend_y - 8))
    
def draw_neural_network(panel_rect, nn, inputs):
    if not hasattr(nn, 'h') or not hasattr(nn, 'probs'):
        # If forward hasn't been called yet, run a forward pass
        nn.forward(inputs)
    
    h = nn.h
    probs = nn.probs
    
    panel_x, panel_y, panel_w, panel_h = panel_rect
    
    # Adjust node positions for the smaller panel
    input_x = panel_x + 40
    hidden_x = panel_x + panel_w // 2
    output_x = panel_x + panel_w - 40
    
    n_input = nn.input_size
    n_hidden = nn.hidden_size
    n_output = nn.output_size
    
    spacing_input = panel_h / (n_input + 1)
    spacing_hidden = panel_h / (n_hidden + 1)
    spacing_output = panel_h / (n_output + 1)
    
    input_neurons = [(input_x, panel_y + (i + 1) * spacing_input) for i in range(n_input)]
    hidden_neurons = [(hidden_x, panel_y + (i + 1) * spacing_hidden) for i in range(n_hidden)]
    output_neurons = [(output_x, panel_y + (i + 1) * spacing_output) for i in range(n_output)]
    
    # Draw panel background
    pygame.draw.rect(screen, WHITE, panel_rect)
    pygame.draw.rect(screen, BLACK, panel_rect, 1)
    
    # Draw title
    title = FONT_SMALL.render("Neural Network", True, BLACK)
    screen.blit(title, (panel_x + (panel_w - title.get_width()) // 2, panel_y - 20))
    
    # Draw connections (simplified to reduce visual clutter)
    for i, inp_pos in enumerate(input_neurons):
        for j, hid_pos in enumerate(hidden_neurons):
            weight = nn.w1[i, j]
            if abs(weight) > 0.4:  # Only show stronger connections
                color = BLUE if weight > 0 else RED
                width = max(1, int(abs(weight) * 2))
                pygame.draw.line(screen, color, inp_pos, hid_pos, width)
    
    for i, hid_pos in enumerate(hidden_neurons):
        for j, out_pos in enumerate(output_neurons):
            weight = nn.w2[i, j]
            if abs(weight) > 0.4:  # Only show stronger connections
                color = BLUE if weight > 0 else RED
                width = max(1, int(abs(weight) * 2))
                pygame.draw.line(screen, color, hid_pos, out_pos, width)
    
    # Draw nodes with smaller size
    node_size = 4
    
    # Draw input neurons
    for i, pos in enumerate(input_neurons):
        if i < NUM_RAYS:
            activation = inputs[i]
            color_val = int(activation * 255)
            pygame.draw.circle(screen, (color_val, color_val, 0), pos, node_size)
        else:
            activation = inputs[i]
            color_val = int(abs(activation) * 255)
            pygame.draw.circle(screen, (0, color_val, color_val), pos, node_size)
        pygame.draw.circle(screen, BLACK, pos, node_size, 1)
    
    # Draw hidden neurons
    for i, pos in enumerate(hidden_neurons):
        activation = h[i]
        color_val = int(min(activation, 1) * 255)
        pygame.draw.circle(screen, (0, color_val, 0), pos, node_size)
        pygame.draw.circle(screen, BLACK, pos, node_size, 1)
    
    # Draw output neurons with smaller labels
    output_labels = ["L", "S", "R"]  # Only steering labels
    for i, pos in enumerate(output_neurons):
        activation = probs[i]
        color_val = int(activation * 255)
        pygame.draw.circle(screen, (0, 0, color_val), pos, node_size + 2)
        pygame.draw.circle(screen, BLACK, pos, node_size + 2, 1)
        
        # Add output labels
        label = output_labels[i]
        text = FONT_SMALL.render(label, True, BLACK)
        screen.blit(text, (pos[0] + 8, pos[1] - 8))

# Modified draw_car function
def draw_car(car, is_best=False):
    if is_best:
        if car.alive:
            color = BRIGHT_GREEN  # Leading car is alive
        else:
            color = DARK_GREEN   # Leading car is dead
    elif not car.alive:
        color = (100, 100, 100)  # Grey for dead non-leading cars
    else:
        color = RED              # Red for alive non-leading cars
    
    # Draw the car body (example implementation, adjust as needed)
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

def run_simulation():
    global walls, checkpoints, road_polygons, start_pos, start_angle
    generation_frame_count = 0

    # Select track from menu
    track_type = menu()
    set_track(track_type)

    # Initialize first generation of cars with random NNs
    cars = [Car(NeuralNetwork()) for _ in range(NUM_CARS)]
    is_first_frame = True  # NEW: Flag to indicate the first frame of a generation
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
        # Calculate delta_time at the start of the loop
        delta_time = clock.tick(60) / 1000.0  # Convert milliseconds to seconds
        generation_frame_count += 1

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
        
        # NEW: Update alive cars only if not the first frame
        if not is_first_frame:
            alive_count = 0
            for car in cars:
                if car.alive:
                    car.decide_actions()
                    car.update(delta_time)
                    alive_count += 1
        else:
            alive_count = NUM_CARS  # All cars are alive on the first frame

        # Find the best car among all cars (alive or dead)
        if cars:
            best_car = max(cars, key=lambda car: car.fitness)
            best_fitness_current = best_car.fitness
        else:
            best_car = None
            best_fitness_current = -float('inf')
        
        # Draw cars with highlighting for the best car
        if show_all_cars:
            for car in cars:
                draw_car(car, is_best=(car == best_car))
        elif best_car:
            draw_car(best_car, is_best=True)
        
        # NEW: After drawing the first frame, reset the flag
        if is_first_frame:
            is_first_frame = False
        
        # Update all-time best NN if we have a new champion
        if best_fitness_current > best_fitness:
            best_fitness = best_fitness_current
            if best_car:
                best_nn = best_car.nn.clone()
        
        # Draw neural network visualization for best car
        if best_car:
            # Prepare inputs for visualization
            ray_distances = best_car.cast_rays()
            normalized_steering = best_car.steering_angle / MAX_STEERING_ANGLE
            checkpoint_progress = best_car.current_checkpoint / len(checkpoints)
            checkpoint_angle = best_car.calculate_angle_to_next_checkpoint()

            inputs = np.concatenate([
                ray_distances, 
                [normalized_steering, checkpoint_progress, checkpoint_angle]
            ])          
            # Draw neural network visualization
            draw_neural_network(NN_PANEL_RECT, best_car.nn, inputs)
            
            # Draw performance graph
            draw_performance_graph(GRAPH_RECT, generation_stats)
        
        # Display stats
        stats = [
            f"FPS: {clock.get_fps():.1f}",
            f"Generation: {generation}",
            f"Time: {(generation_frame_count / 60) * 2:.2f}s/{(GENERATION_FRAMES / 60) * 2:.2f}s ({(generation_frame_count / GENERATION_FRAMES) * 100:.0f}%)",
            f"Cars Alive: {alive_count}/{NUM_CARS}",
            f"Best Fitness: {best_fitness:.2f}",
            f"Current Best: {best_fitness_current:.2f}",
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
            screen.blit(text_surface, (10, 10 + i * 24))
        
        # Check if generation time is up or all cars are dead
        current_time = pygame.time.get_ticks()
        generation_elapsed = current_time - generation_start_time
        
        if generation_frame_count >= GENERATION_FRAMES or alive_count == 0:
            # Calculate best_fitness_current as the max of ALL cars (alive or dead)
            best_fitness_current = max(car.fitness for car in cars)
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
            generation_frame_count = 0
            is_first_frame = True  # NEW: Reset flag for the new generation
            
            # Display generation change message
            gen_msg = f"Generation {generation} started!"
            gen_text = FONT_MEDIUM.render(gen_msg, True, BLUE)
            screen.blit(gen_text, (SIM_WIDTH // 2 - gen_text.get_width() // 2, 50))
            pygame.display.flip()
            pygame.time.delay(1000)  # Pause briefly to show generation change
        
        pygame.display.flip()
        delta_time = clock.tick(60) / 1000.0  # Convert milliseconds to seconds

    pygame.quit()

if __name__ == "__main__":
    run_simulation()