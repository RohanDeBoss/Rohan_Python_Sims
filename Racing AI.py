import pygame
import math
import numpy as np
import random
import sys

# Global Parameters
NUM_CARS = 70
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.5
CAR_SPEED = 5  # Initial speed in units per frame
NUM_RAYS = 8
VISION_CONE_ANGLE = math.pi / 2
MIN_SPEED = 3
MAX_SPEED = 8
GENERATION_DURATION = 10 * 1000  # 20 seconds in milliseconds
CAR_WIDTH = 20
CAR_HEIGHT = 10
SIM_WIDTH = 800
SIM_HEIGHT = 600
NN_PANEL_WIDTH = 300
NN_PANEL_HEIGHT = 300
WIDTH = SIM_WIDTH + NN_PANEL_WIDTH
HEIGHT = SIM_HEIGHT
NN_PANEL_RECT = (SIM_WIDTH, (HEIGHT - NN_PANEL_HEIGHT) // 2, NN_PANEL_WIDTH, NN_PANEL_HEIGHT)

# New Physics Parameters
WHEELBASE = 5            # Distance between front and rear axles for bicycle model
MAX_STEERING_ANGLE = 0.3  # Maximum steering angle in radians (about 17 degrees)
ACCELERATION = 0.05       # Acceleration/deceleration per frame

# Pygame Initialization
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Neural Network Racing Game")
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
    for i in range(0, num_points, 10):
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
        for i in range(10):
            angle = 2 * math.pi * i / 10
            p_inner = (center[0] + inner_radius * math.cos(angle), center[1] + inner_radius * math.sin(angle))
            p_outer = (center[0] + outer_radius * math.cos(angle), center[1] + outer_radius * math.sin(angle))
            checkpoints.append([p_inner, p_outer])
        start_pos = (center[0] + (inner_radius + outer_radius) / 2, center[1])
        start_angle = math.pi / 2
    elif track_type == "medium":
        outer_a = 250
        outer_b = 200
        inner_a = 150  # Widened from 200 to 150
        inner_b = 100  # Widened from 150 to 100
        num_segments = 40
        outer_walls = generate_ellipse_segments(center, outer_a, outer_b, num_segments)
        inner_walls = generate_ellipse_segments(center, inner_a, inner_b, num_segments)
        walls = outer_walls + inner_walls
        outer_points = [wall[0] for wall in outer_walls]
        inner_points = [wall[0] for wall in inner_walls]
        track_points = outer_points + inner_points[::-1]
        road_polygons = [track_points]
        checkpoints.clear()
        for i in range(10):
            angle = 2 * math.pi * i / 10
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

# Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.w1 = np.random.randn(NUM_RAYS, 16)
        self.b1 = np.zeros(16)
        self.w2 = np.random.randn(16, 6)
        self.b2 = np.zeros(6)
    
    def forward(self, x):
        h = np.maximum(0, np.dot(x, self.w1) + self.b1)
        o = np.dot(h, self.w2) + self.b2
        exp_o = np.exp(o - np.max(o))
        probs = exp_o / np.sum(exp_o)
        return probs
    
    def mutate(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        new_nn = self.clone()
        new_nn.w1 += np.random.randn(*self.w1.shape) * mutation_strength * (np.random.rand(*self.w1.shape) < mutation_rate)
        new_nn.b1 += np.random.randn(*self.b1.shape) * mutation_strength * (np.random.rand(*self.b1.shape) < mutation_rate)
        new_nn.w2 += np.random.randn(*self.w2.shape) * mutation_strength * (np.random.rand(*self.w2.shape) < mutation_rate)
        new_nn.b2 += np.random.randn(*self.b2.shape) * mutation_strength * (np.random.rand(*self.b2.shape) < mutation_rate)
        return new_nn

    def clone(self):
        clone_nn = NeuralNetwork()
        clone_nn.w1 = np.copy(self.w1)
        clone_nn.b1 = np.copy(self.b1)
        clone_nn.w2 = np.copy(self.w2)
        clone_nn.b2 = np.copy(self.b2)
        return clone_nn

# Car Class with Improved Physics
class Car:
    def __init__(self, nn):
        self.nn = nn
        self.x, self.y = start_pos
        self.angle = start_angle
        self.speed = CAR_SPEED
        self.steering_angle = 0  # Current steering angle in radians
        self.acceleration = 0    # Current acceleration per frame
        self.alive = True
        self.fitness = 0
        self.current_checkpoint = 0
        self.prev_x = self.x
        self.prev_y = self.y

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
            distances.append(dist / 200)
        return np.array(distances)

    def decide_actions(self):
        inputs = self.cast_rays()
        outputs = self.nn.forward(inputs)
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
            # Update speed with acceleration
            self.speed += self.acceleration
            self.speed = max(MIN_SPEED, min(MAX_SPEED, self.speed))

            # Update angle using bicycle model (turning rate depends on speed)
            dtheta = (self.speed / WHEELBASE) * math.tan(self.steering_angle)
            self.angle += dtheta
            self.angle %= (2 * math.pi)

            # Update position
            self.prev_x = self.x
            self.prev_y = self.y
            self.x += self.speed * math.cos(self.angle)
            self.y += self.speed * math.sin(self.angle)

            # Check for collisions and checkpoints
            if self.check_collision():
                self.alive = False
            if self.check_checkpoint():
                self.current_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
                self.fitness += 1

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
        pygame.draw.line(screen, GREEN, cp[0], cp[1], 2)

def draw_neural_network(panel_rect, nn, inputs):
    h = np.maximum(0, np.dot(inputs, nn.w1) + nn.b1)
    o = np.dot(h, nn.w2) + nn.b2
    exp_o = np.exp(o - np.max(o))
    probs = exp_o / np.sum(exp_o)
    panel_x, panel_y, panel_w, panel_h = panel_rect
    input_x = panel_x + 50
    hidden_x = panel_x + panel_w // 2
    output_x = panel_x + panel_w - 50
    n_input = NUM_RAYS
    n_hidden = 16
    n_output = 6
    spacing_input = panel_h / (n_input + 1)
    spacing_hidden = panel_h / (n_hidden + 1)
    spacing_output = panel_h / (n_output + 1)
    input_neurons = [(input_x, panel_y + (i + 1) * spacing_input) for i in range(n_input)]
    hidden_neurons = [(hidden_x, panel_y + (i + 1) * spacing_hidden) for i in range(n_hidden)]
    output_neurons = [(output_x, panel_y + (i + 1) * spacing_output) for i in range(n_output)]
    for i, inp_pos in enumerate(input_neurons):
        for j, hid_pos in enumerate(hidden_neurons):
            weight = nn.w1[i, j]
            color = BLUE if weight > 0 else RED
            pygame.draw.line(screen, color, inp_pos, hid_pos, max(1, int(abs(weight) * 2)))
    for i, hid_pos in enumerate(hidden_neurons):
        for j, out_pos in enumerate(output_neurons):
            weight = nn.w2[i, j]
            color = BLUE if weight > 0 else RED
            pygame.draw.line(screen, color, hid_pos, out_pos, max(1, int(abs(weight) * 2)))
    for i, pos in enumerate(input_neurons):
        activation = inputs[i]
        color_val = int(activation * 255)
        pygame.draw.circle(screen, (color_val, color_val, 0), pos, 6)
        pygame.draw.circle(screen, BLACK, pos, 6, 1)
    for i, pos in enumerate(hidden_neurons):
        activation = h[i]
        color_val = int(min(activation, 1) * 255)
        pygame.draw.circle(screen, (0, color_val, 0), pos, 6)
        pygame.draw.circle(screen, BLACK, pos, 6, 1)
    for i, pos in enumerate(output_neurons):
        activation = probs[i]
        color_val = int(activation * 255)
        pygame.draw.circle(screen, (0, 0, color_val), pos, 6)
        pygame.draw.circle(screen, BLACK, pos, 6, 1)

def draw_car(car):
    w, h = CAR_WIDTH, CAR_HEIGHT
    local_corners = [(-w/2, -h/2), (w/2, -h/2), (w/2, h/2), (-w/2, h/2)]
    rotated = []
    for (x, y) in local_corners:
        rx = x * math.cos(car.angle) - y * math.sin(car.angle)
        ry = x * math.sin(car.angle) + y * math.cos(car.angle)
        rotated.append((car.x + rx, car.y + ry))
    pygame.draw.polygon(screen, RED, rotated)
    pygame.draw.polygon(screen, BLACK, rotated, 1)

# Main Simulation
def main():
    track_type = menu()
    set_track(track_type)
    print(f"Track selected: {track_type}")
    generation = 0
    cars = [Car(NeuralNetwork()) for _ in range(NUM_CARS)]
    best_fitness = 0
    average_fitness = 0
    best_car = cars[0]
    generation_start_time = pygame.time.get_ticks()

    running = True
    while running:
        current_time = pygame.time.get_ticks()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False

        for car in cars:
            if car.alive:
                car.decide_actions()
                car.update()

        if all(not car.alive for car in cars) or (current_time - generation_start_time) > GENERATION_DURATION:
            generation += 1
            fitnesses = [car.fitness for car in cars]
            best_idx = np.argmax(fitnesses)
            best_car = cars[best_idx]
            best_fitness = fitnesses[best_idx]
            average_fitness = np.mean(fitnesses)
            print(f"Generation {generation}: Avg Fitness = {average_fitness:.2f}, Best = {best_fitness}")
            new_cars = [Car(best_car.nn.clone())]
            for _ in range(NUM_CARS - 1):
                new_cars.append(Car(best_car.nn.mutate()))
            cars = new_cars
            generation_start_time = pygame.time.get_ticks()

        screen.fill(WHITE)
        pygame.draw.rect(screen, WHITE, (0, 0, SIM_WIDTH, SIM_HEIGHT))
        draw_track()
        for car in cars:
            if car.alive:
                draw_car(car)
                rays = car.cast_rays()
                angle_start = car.angle - VISION_CONE_ANGLE / 2
                delta = VISION_CONE_ANGLE / (NUM_RAYS - 1)
                for i in range(NUM_RAYS):
                    ray_angle = angle_start + i * delta
                    dx = math.cos(ray_angle)
                    dy = math.sin(ray_angle)
                    ray_length = rays[i] * 200
                    end_x = car.x + dx * ray_length
                    end_y = car.y + dy * ray_length
                    pygame.draw.line(screen, BLUE, (car.x, car.y), (end_x, end_y), 1)
        screen.blit(FONT_MEDIUM.render(f"Gen: {generation}", True, BLACK), (10, 10))
        screen.blit(FONT_MEDIUM.render(f"Avg Fitness: {average_fitness:.2f}", True, BLACK), (10, 50))
        screen.blit(FONT_MEDIUM.render(f"Best Fitness: {best_fitness}", True, BLACK), (10, 90))
        best_inputs = best_car.cast_rays()
        draw_neural_network(NN_PANEL_RECT, best_car.nn, best_inputs)
        fps = clock.get_fps()
        fps_text = FONT_SMALL.render(f"FPS: {fps:.1f}", True, BLACK)
        screen.blit(fps_text, (WIDTH - fps_text.get_width() - 10, 10))
        pygame.display.flip()
        clock.tick(60)
    pygame.quit()

if __name__ == "__main__":
    main()