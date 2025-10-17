import pygame
import math
import numpy as np
import random
import sys
from collections import deque
import os
import glob # <<< NEW FEATURE: To find all saved AI files >>>

# Global Parameters
NUM_CARS = 75
MUTATION_RATE = 0.1
MUTATION_STRENGTH = 0.42

NUM_RAYS = 8
VISION_CONE_ANGLE = math.pi * 0.8

CONSTANT_SPEED = 320
GENERATION_DURATION = 15 #20 was too long

CAR_WIDTH = 22
CAR_HEIGHT = 11

# Simulation dimensions
SIM_WIDTH = 850
SIM_HEIGHT = 600
HEIGHT = SIM_HEIGHT
UI_WIDTH = 320
WIDTH = SIM_WIDTH + UI_WIDTH

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

BRIGHT_GREEN = (0, 255, 0)
DARK_GREEN = (0, 100, 0)

# Enhanced Physics Parameters
WHEELBASE = 5
MAX_STEERING_ANGLE = 0.2
CRASH_PENALTY = 15

# Pygame Initialization
pygame.init()
pygame.font.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Enhanced Neural Network Racing Simulation")
clock = pygame.time.Clock()
FONT_SMALL = pygame.font.Font(None, 24)
FONT_MEDIUM = pygame.font.Font(None, 36)
FONT_LARGE = pygame.font.Font(None, 48)

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
        theta1 = 2 * math.pi * i / num_segments; theta2 = 2 * math.pi * (i + 1) / num_segments
        p1 = (cx + radius * math.cos(theta1), cy + radius * math.sin(theta1))
        p2 = (cx + radius * math.cos(theta2), cy + radius * math.sin(theta2))
        segments.append([p1, p2])
    return segments
def generate_ellipse_segments(center, a, b, num_segments):
    segments = []
    cx, cy = center
    for i in range(num_segments):
        t1 = 2 * math.pi * i / num_segments; t2 = 2 * math.pi * (i + 1) / num_segments
        p1 = (cx + a * math.cos(t1), cy + b * math.sin(t1))
        p2 = (cx + a * math.cos(t2), cy + b * math.sin(t2))
        segments.append([p1, p2])
    return segments
def generate_square_track(center, size, track_width):
    cx, cy = center
    length_multiplier, skinnier_factor = 1.5, 0.95; adjusted_track_width = track_width * skinnier_factor
    outer_width, outer_height = size * length_multiplier, size; inner_width, inner_height = outer_width - adjusted_track_width, outer_height - adjusted_track_width
    outer_corners = [(cx - outer_width, cy - outer_height), (cx + outer_width, cy - outer_height), (cx + outer_width, cy + outer_height), (cx - outer_width, cy + outer_height)]
    inner_corners = [(cx - inner_width, cy - inner_height), (cx + inner_width, cy - inner_height), (cx + inner_width, cy + inner_height), (cx - inner_width, cy + inner_height)]
    points_per_short_edge, points_per_long_edge = 20, int(20 * length_multiplier)
    outer_points, inner_points = [], []
    for j in range(points_per_long_edge): t = j/points_per_long_edge; outer_points.append((outer_corners[0][0]+t*(outer_corners[1][0]-outer_corners[0][0]), outer_corners[0][1]))
    for j in range(points_per_short_edge): t = j/points_per_short_edge; outer_points.append((outer_corners[1][0], outer_corners[1][1]+t*(outer_corners[2][1]-outer_corners[1][1])))
    for j in range(points_per_long_edge): t = j/points_per_long_edge; outer_points.append((outer_corners[2][0]-t*(outer_corners[2][0]-outer_corners[3][0]), outer_corners[2][1]))
    for j in range(points_per_short_edge): t = j/points_per_short_edge; outer_points.append((outer_corners[3][0], outer_corners[3][1]-t*(outer_corners[3][1]-outer_corners[0][1])))
    for j in range(points_per_long_edge): t = j/points_per_long_edge; inner_points.append((inner_corners[0][0]+t*(inner_corners[1][0]-inner_corners[0][0]), inner_corners[0][1]))
    for j in range(points_per_short_edge): t = j/points_per_short_edge; inner_points.append((inner_corners[1][0], inner_corners[1][1]+t*(inner_corners[2][1]-inner_corners[1][1])))
    for j in range(points_per_long_edge): t = j/points_per_long_edge; inner_points.append((inner_corners[2][0]-t*(inner_corners[2][0]-inner_corners[3][0]), inner_corners[2][1]))
    for j in range(points_per_short_edge): t = j/points_per_short_edge; inner_points.append((inner_corners[3][0], inner_corners[3][1]-t*(inner_corners[3][1]-inner_corners[0][1])))
    centerline_points = [((op[0]+ip[0])/2, (op[1]+ip[1])/2) for op, ip in zip(outer_points, inner_points)]
    walls = []
    for i in range(len(outer_points)): walls.append([outer_points[i], outer_points[(i + 1) % len(outer_points)]])
    for i in range(len(inner_points)): walls.append([inner_points[i], inner_points[(i + 1) % len(inner_points)]])
    road_polygons = [outer_points + inner_points[::-1]]
    checkpoints = [[inner_points[i], outer_points[i]] for i in range(0, len(outer_points), len(outer_points)//25)]
    start_pos = centerline_points[0]
    p0, p1 = centerline_points[0], centerline_points[1]
    start_angle = math.atan2(p1[1]-p0[1], p1[0]-p0[0])
    return walls, road_polygons, checkpoints, start_pos, start_angle

def set_track(track_type):
    global center, walls, checkpoints, road_polygons, start_pos, start_angle
    if track_type == "easy":
        center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2)
        outer_radius, inner_radius, num_segments = 250, 150, 40
        outer_walls, inner_walls = generate_circle_segments(center, outer_radius, num_segments), generate_circle_segments(center, inner_radius, num_segments)
        walls = outer_walls + inner_walls
        outer_points, inner_points = [w[0] for w in outer_walls], [w[0] for w in inner_walls]
        road_polygons = [outer_points + inner_points[::-1]]
        checkpoints.clear()
        for i in range(20): angle = 2*math.pi*i/20; checkpoints.append([(center[0]+inner_radius*math.cos(angle), center[1]+inner_radius*math.sin(angle)),(center[0]+outer_radius*math.cos(angle), center[1]+outer_radius*math.sin(angle))])
        start_pos = (center[0] + (inner_radius + outer_radius) / 2, center[1]); start_angle = math.pi / 2
    elif track_type == "medium":
        center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2)
        outer_a, outer_b, inner_a, inner_b, num_segments = 250, 200, 170, 120, 40
        outer_walls, inner_walls = generate_ellipse_segments(center, outer_a, outer_b, num_segments), generate_ellipse_segments(center, inner_a, inner_b, num_segments)
        walls = outer_walls + inner_walls
        outer_points, inner_points = [w[0] for w in outer_walls], [w[0] for w in inner_walls]
        road_polygons = [outer_points + inner_points[::-1]]
        checkpoints.clear()
        for i in range(20): angle = 2*math.pi*i/20; checkpoints.append([(center[0]+inner_a*math.cos(angle), center[1]+inner_b*math.sin(angle)),(center[0]+outer_a*math.cos(angle), center[1]+outer_b*math.sin(angle))])
        start_pos = (center[0] + (inner_a + outer_a) / 2, center[1]); start_angle = math.pi / 2
    elif track_type == "hard":
        hard_center = (SIM_WIDTH // 2 + 50, SIM_HEIGHT // 2 + 50)
        square_size, track_width = 175, 70
        walls, road_polygons, checkpoints, start_pos, start_angle = generate_square_track(hard_center, square_size, track_width)

# Menu Systems
def track_select_menu():
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: return "easy"
                elif event.key == pygame.K_2: return "medium"
                elif event.key == pygame.K_3: return "hard"
                elif event.key == pygame.K_4: return "impossible"
        screen.fill(WHITE)
        title_text = FONT_MEDIUM.render("Select Track", True, BLACK)
        screen.blit(title_text, (WIDTH / 2 - title_text.get_width() / 2, 100))
        option1 = FONT_SMALL.render("1 - Easy", True, BLACK)
        option2 = FONT_SMALL.render("2 - Medium", True, BLACK)
        option3 = FONT_SMALL.render("3 - Hard", True, BLACK)
        option4 = FONT_SMALL.render("4 - Mixed Tracks (Random)", True, BLACK)
        screen.blit(option1, (WIDTH/2 - option1.get_width()/2, 200)); screen.blit(option2, (WIDTH/2 - option2.get_width()/2, 240))
        screen.blit(option3, (WIDTH/2 - option3.get_width()/2, 280)); screen.blit(option4, (WIDTH/2 - option4.get_width()/2, 320))
        pygame.display.flip(); clock.tick(60)

# <<< NEW FEATURE: Menu to select a saved AI to load >>>
def load_ai_menu():
    saved_ais = sorted(glob.glob("saved_ais/*.npz"))
    if not saved_ais: return None # No AI saved, go back to main menu
    
    page_size = 8
    page = 0
    while True:
        start_index = page * page_size
        end_index = start_index + page_size
        current_page_ais = saved_ais[start_index:end_index]

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: return None # Go back
                if event.key == pygame.K_RIGHT and end_index < len(saved_ais): page += 1
                if event.key == pygame.K_LEFT and page > 0: page -= 1
                if pygame.K_1 <= event.key <= pygame.K_9:
                    selection = event.key - pygame.K_1
                    if selection < len(current_page_ais):
                        return current_page_ais[selection]

        screen.fill(WHITE)
        title_text = FONT_MEDIUM.render("Select Saved AI to Test", True, BLACK)
        screen.blit(title_text, (WIDTH / 2 - title_text.get_width() / 2, 50))
        
        for i, ai_path in enumerate(current_page_ais):
            ai_name = os.path.basename(ai_path).replace('.npz', '')
            text = FONT_SMALL.render(f"{i+1} - {ai_name}", True, BLACK)
            screen.blit(text, (WIDTH / 2 - text.get_width() / 2, 150 + i * 40))
            
        page_text = FONT_SMALL.render(f"Page {page+1}/{math.ceil(len(saved_ais)/page_size)} (Use Left/Right Arrows)", True, GREY)
        screen.blit(page_text, (WIDTH / 2 - page_text.get_width() / 2, 550))

        pygame.display.flip(); clock.tick(60)

def main_menu():
    if not os.path.exists("saved_ais"): os.makedirs("saved_ais")
    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_1: return "TRAIN"
                elif event.key == pygame.K_2:
                    if len(glob.glob("saved_ais/*.npz")) > 0: return "TEST"
                    else: print("No saved AI found! Please train an AI first.")
        
        screen.fill(WHITE)
        title_text = FONT_LARGE.render("Racing AI Simulation", True, BLACK)
        screen.blit(title_text, (WIDTH/2 - title_text.get_width()/2, 100))
        option1 = FONT_MEDIUM.render("1 - Train New AI", True, BLACK)
        screen.blit(option1, (WIDTH/2 - option1.get_width()/2, 250))
        
        test_color = BLACK if len(glob.glob("saved_ais/*.npz")) > 0 else GREY
        option2 = FONT_MEDIUM.render("2 - Test Saved AI", True, test_color)
        screen.blit(option2, (WIDTH/2 - option2.get_width()/2, 310))
        pygame.display.flip(); clock.tick(60)

# Enhanced Neural Network Class
class NeuralNetwork:
    def __init__(self):
        self.input_size = 8 + 3; self.hidden_size = 14; self.output_size = 3
        self.w1 = np.random.randn(self.input_size, self.hidden_size) * np.sqrt(2.0/self.input_size)
        self.b1 = np.zeros(self.hidden_size)
        self.w2 = np.random.randn(self.hidden_size, self.output_size) * np.sqrt(2.0/self.hidden_size)
        self.b2 = np.zeros(self.output_size)

    # <<< NEW FEATURE: Smarter save and load methods >>>
    def save(self, filename):
        np.savez(filename, w1=self.w1, b1=self.b1, w2=self.w2, b2=self.b2)
        print(f"Neural network saved to {filename}")

    @staticmethod
    def load(filename):
        data = np.load(filename)
        nn = NeuralNetwork()
        nn.w1, nn.b1, nn.w2, nn.b2 = data['w1'], data['b1'], data['w2'], data['b2']
        print(f"Neural network loaded from {filename}")
        return nn

    def forward(self, x):
        self.h = np.maximum(0, np.dot(x, self.w1) + self.b1)
        o = np.dot(self.h, self.w2) + self.b2
        exp_o = np.exp(o - np.max(o)); self.probs = exp_o / np.sum(exp_o)
        return self.probs
    def mutate(self, mutation_rate=MUTATION_RATE, mutation_strength=MUTATION_STRENGTH):
        new_nn = self.clone()
        new_nn.w1 += np.random.randn(*self.w1.shape)*mutation_strength*(np.random.rand(*self.w1.shape)<mutation_rate)
        new_nn.b1 += np.random.randn(*self.b1.shape)*mutation_strength*(np.random.rand(*self.b1.shape)<mutation_rate)
        new_nn.w2 += np.random.randn(*self.w2.shape)*mutation_strength*(np.random.rand(*self.w2.shape)<mutation_rate)
        new_nn.b2 += np.random.randn(*self.b2.shape)*mutation_strength*(np.random.rand(*self.b2.shape)<mutation_rate)
        return new_nn
    def crossover(self, other):
        child = NeuralNetwork()
        mask1 = np.random.rand(*self.w1.shape) < 0.5; mask2 = np.random.rand(*self.w2.shape) < 0.5
        child.w1 = np.where(mask1, self.w1, other.w1); child.b1 = np.where(np.random.rand(*self.b1.shape) < 0.5, self.b1, other.b1)
        child.w2 = np.where(mask2, self.w2, other.w2); child.b2 = np.where(np.random.rand(*self.b2.shape) < 0.5, self.b2, other.b2)
        return child
    def clone(self):
        clone_nn = NeuralNetwork(); clone_nn.w1=np.copy(self.w1); clone_nn.b1=np.copy(self.b1); clone_nn.w2=np.copy(self.w2); clone_nn.b2=np.copy(self.b2)
        return clone_nn

# Car Class (Unchanged)
class Car:
    def __init__(self, nn):
        self.last_checkpoint_time = pygame.time.get_ticks(); self.nn = nn
        self.x, self.y = start_pos; self.angle = start_angle
        self.speed = CONSTANT_SPEED; self.steering_angle = 0
        self.alive = True; self.fitness = 0; self.raw_fitness = 0
        self.distance_traveled = 0; self.checkpoint_times = {}
        self.current_checkpoint = 0; self.prev_checkpoint = 0; self.lap_count = 0
        self.prev_x = self.x; self.prev_y = self.y; self.idle_time = 0
        self.avg_speed = deque(maxlen=60); self.wrong_way = False
        self.position_history = deque(maxlen=60); self.crashes = 0
        self.start_time = pygame.time.get_ticks()
        for _ in range(10): self.position_history.append((self.x, self.y))
    def calculate_angle_to_next_checkpoint(self):
        cp = checkpoints[self.current_checkpoint]
        cp_x = (cp[0][0] + cp[1][0])/2; cp_y = (cp[0][1] + cp[1][1])/2
        dx = cp_x - self.x; dy = cp_y - self.y; cp_angle = math.atan2(dy, dx)
        rel_angle = (cp_angle - self.angle) % (2*math.pi)
        if rel_angle > math.pi: rel_angle -= 2*math.pi
        return rel_angle/math.pi
    def cast_rays(self):
        distances = []
        angle_start = self.angle - VISION_CONE_ANGLE/2; delta = VISION_CONE_ANGLE/(NUM_RAYS - 1)
        for i in range(NUM_RAYS):
            ray_angle = angle_start + i*delta
            dx = math.cos(ray_angle); dy = math.sin(ray_angle); min_t = float('inf')
            for wall in walls:
                ax, ay = wall[0]; bx, by = wall[1]; D = dx*(by-ay)-dy*(bx-ax)
                if abs(D) > 1e-6:
                    t = ((ay - self.y)*(bx-ax) - (ax - self.x)*(by-ay))/D
                    s = (dx*(ay - self.y) - dy*(ax - self.x))/D
                    if t >= 0 and 0 <= s <= 1: min_t = min(min_t, t)
            dist = min(min_t, 200) if min_t < float('inf') else 200
            distances.append(dist/200)
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
            self.prev_x, self.prev_y = self.x, self.y
            self.speed = CONSTANT_SPEED; self.avg_speed.append(self.speed)
            dtheta = (self.speed/WHEELBASE)*math.tan(self.steering_angle)*delta_time
            self.angle = (self.angle + dtheta) % (2*math.pi)
            self.x += self.speed*math.cos(self.angle)*delta_time
            self.y += self.speed*math.sin(self.angle)*delta_time
            self.position_history.append((self.x, self.y))
            self.distance_traveled += math.hypot(self.x-self.prev_x, self.y-self.prev_y)
            if self.check_collision(): self.alive, self.crashes, self.fitness = False, self.crashes+1, self.fitness-CRASH_PENALTY
            self.prev_checkpoint = self.current_checkpoint
            if self.check_checkpoint():
                self.current_checkpoint = (self.current_checkpoint + 1) % len(checkpoints)
                self.last_checkpoint_time = pygame.time.get_ticks()
                self.fitness, self.raw_fitness = self.fitness+10, self.raw_fitness+1
                if self.current_checkpoint == 0 and self.prev_checkpoint == len(checkpoints)-1: self.lap_count, self.fitness = self.lap_count+1, self.fitness+50
            if pygame.time.get_ticks() - self.last_checkpoint_time > 5000: self.alive, self.fitness = False, self.fitness-10
            if len(self.position_history) >= 60:
                dist_moved = math.hypot(self.position_history[-1][0] - self.position_history[0][0], self.position_history[-1][1] - self.position_history[0][1])
                self.idle_time = self.idle_time+1 if dist_moved < 30 else 0
                if self.idle_time > 120: self.alive, self.fitness = False, self.fitness-20
    def check_collision(self):
        car_path = [(self.prev_x, self.prev_y), (self.x, self.y)]
        for wall in walls:
            if self.intersects(car_path, wall): return True
        return False
    def check_checkpoint(self): return self.intersects([(self.prev_x, self.prev_y), (self.x, self.y)], checkpoints[self.current_checkpoint])
    def intersects(self, s1, s2): p1,q1=s1; p2,q2=s2; o1,o2,o3,o4 = self.orientation(p1,q1,p2), self.orientation(p1,q1,q2), self.orientation(p2,q2,p1), self.orientation(p2,q2,q1); return o1!=o2 and o3!=o4
    def orientation(self, p, q, r): val = (q[1]-p[1])*(r[0]-q[0])-(q[0]-p[0])*(r[1]-q[1]); return 0 if abs(val)<1e-6 else (1 if val>0 else 2)

# Drawing Functions (Unchanged)
def draw_track():
    for poly in road_polygons: pygame.draw.polygon(screen, GREY, poly)
    for wall in walls: pygame.draw.line(screen, BLACK, wall[0], wall[1], 2)
    for cp in checkpoints: pygame.draw.line(screen, GREEN, cp[0], cp[1], 1)
def draw_performance_graph(rect, stats):
    x, y, w, h = rect; pygame.draw.rect(screen, WHITE, rect); pygame.draw.rect(screen, BLACK, rect, 1)
    title = FONT_SMALL.render("Performance History", True, BLACK); screen.blit(title, (x + (w - title.get_width()) // 2, y + 5))
    if len(stats) < 2: msg = FONT_SMALL.render("Collecting data...", True, (100, 100, 100)); screen.blit(msg, (x + (w-msg.get_width())//2, y+h//2)); return
    y_min=0.0; gens=[s['generation'] for s in stats]; best_fit,avg_fit = [],[]
    for s in stats: best_fit.append(s['best_fitness'] if math.isfinite(s['best_fitness']) else y_min); avg_fit.append(s['avg_fitness'] if math.isfinite(s['avg_fitness']) else y_min)
    valid_best,valid_avg = [f for f in best_fit if math.isfinite(f)],[f for f in avg_fit if math.isfinite(f)]
    y_max = max(max(valid_best), max(valid_avg))*1.1 if valid_best and valid_avg else 1.0; y_max = y_min + 1.0 if y_max <= y_min else y_max
    gx,gy,gw,gh = x+40,y+25,w-50,h-60
    pygame.draw.line(screen, BLACK, (gx, gy), (gx, gy+gh), 1); pygame.draw.line(screen, BLACK, (gx, gy+gh), (gx+gw, gy+gh), 1)
    for i in range(5): frac=i/4.0; val=y_min+frac*(y_max-y_min); y_pos=gy+gh-int(frac*gh); label=FONT_SMALL.render(f"{val:.0f}",True,BLACK); screen.blit(label, (gx-label.get_width()-5, y_pos-label.get_height()//2))
    tot_gens,max_gen = len(gens),gens[-1] if gens else 0
    for i in range(min(tot_gens, 5)):
        idx=i*(tot_gens-1)//4 if tot_gens > 1 else 0; gen_label=gens[idx]; denom = max_gen-gens[0] if gens and max_gen!=gens[0] else 1
        frac=(gen_label-gens[0])/denom; x_pos=gx+int(frac*gw); label=FONT_SMALL.render(f"{gen_label}",True,BLACK); screen.blit(label, (x_pos-label.get_width()//2, gy+gh+5))
    def f_to_y(f): return gy+gh-int(((f-y_min)/(y_max-y_min))*gh) if y_max-y_min>1e-9 and math.isfinite(f) else gy+gh//2
    denom=max_gen-gens[0] if gens and max_gen!=gens[0] else 1
    best_pts = [(gx+int(((g-gens[0])/denom)*gw), f_to_y(best_fit[i])) for i,g in enumerate(gens)]
    if len(best_pts)>1: pygame.draw.lines(screen, BLUE, False, best_pts, 2)
    avg_pts = [(gx+int(((g-gens[0])/denom)*gw), f_to_y(avg_fit[i])) for i,g in enumerate(gens)]
    if len(avg_pts)>1: pygame.draw.lines(screen, (0,200,0), False, avg_pts, 2)
    lx,ly,ll = gx-140,gy+150,20
    pygame.draw.line(screen, BLUE,(lx,ly),(lx+ll,ly),2); best_label=FONT_SMALL.render("Best",True,BLUE); screen.blit(best_label,(lx+ll+8,ly-8))
    ly+=25; pygame.draw.line(screen, (0,200,0),(lx,ly),(lx+ll,ly),2); avg_label=FONT_SMALL.render("Average",True,(0,200,0)); screen.blit(avg_label,(lx+ll+8,ly-8))
def draw_neural_network(rect, nn, inputs):
    if not hasattr(nn, 'h'): nn.forward(inputs)
    h, probs = nn.h, nn.probs
    px, py, pw, ph = rect; ix, hx, ox = px+40, px+pw//2, px+pw-40
    ni,nh,no = nn.input_size,nn.hidden_size,nn.output_size
    si,sh,so = ph/(ni+1),ph/(nh+1),ph/(no+1)
    i_n,h_n,o_n = [(ix,py+(i+1)*si) for i in range(ni)],[(hx,py+(i+1)*sh) for i in range(nh)],[(ox,py+(i+1)*so) for i in range(no)]
    pygame.draw.rect(screen,WHITE,rect); pygame.draw.rect(screen,BLACK,rect,1)
    title=FONT_SMALL.render("Neural Network",True,BLACK); screen.blit(title,(px+(pw-title.get_width())//2,py-20))
    for i,inp in enumerate(i_n):
        for j,hid in enumerate(h_n):
            w=nn.w1[i,j]
            if abs(w)>0.4: color=BLUE if w>0 else RED; width=max(1,int(abs(w)*2)); pygame.draw.line(screen,color,inp,hid,width)
    for i,hid in enumerate(h_n):
        for j,out in enumerate(o_n):
            w=nn.w2[i,j]
            if abs(w)>0.4: color=BLUE if w>0 else RED; width=max(1,int(abs(w)*2)); pygame.draw.line(screen,color,hid,out,width)
    node_size=4
    for i,p in enumerate(i_n):
        act = inputs[i]; color_val = int(act*255 if i<NUM_RAYS else abs(act)*255)
        color = (color_val,color_val,0) if i<NUM_RAYS else (0,color_val,color_val)
        pygame.draw.circle(screen,color,p,node_size); pygame.draw.circle(screen,BLACK,p,node_size,1)
    for i,p in enumerate(h_n): act=h[i]; color_val=int(min(act,1)*255); pygame.draw.circle(screen,(0,color_val,0),p,node_size); pygame.draw.circle(screen,BLACK,p,node_size,1)
    labels = ["L","S","R"]
    for i,p in enumerate(o_n): act=probs[i]; color_val=int(act*255); pygame.draw.circle(screen,(0,0,color_val),p,node_size+2); pygame.draw.circle(screen,BLACK,p,node_size+2,1); text=FONT_SMALL.render(labels[i],True,BLACK); screen.blit(text,(p[0]+8,p[1]-8))
def draw_car(car, is_best=False):
    color = (BRIGHT_GREEN if car.alive else DARK_GREEN) if is_best else (RED if car.alive else (100,100,100))
    w,h=CAR_WIDTH,CAR_HEIGHT; corners=[(-w/2,-h/2),(w/2,-h/2),(w/2,h/2),(-w/2,h/2)]
    rotated = [(car.x+(x*math.cos(car.angle)-y*math.sin(car.angle)), car.y+(x*math.sin(car.angle)+y*math.cos(car.angle))) for x,y in corners]
    pygame.draw.polygon(screen, color, rotated); pygame.draw.polygon(screen, BLACK, rotated, 1)
    fx = car.x+(w/2)*math.cos(car.angle); fy = car.y+(w/2)*math.sin(car.angle)
    pygame.draw.circle(screen, BLACK, (int(fx), int(fy)), 3)
# Genetic Functions (Unchanged)
def tournament_selection(cars, k=5): return max(random.sample(cars, k), key=lambda car: car.fitness)
def elitism(cars, count=5): return [car.nn.clone() for car in sorted(cars, key=lambda x: x.fitness, reverse=True)[:count]]
def select_parents(cars): return tournament_selection(cars)

def run_simulation(mode, saved_ai_path=None):
    if mode == "TRAIN":
        track_type = track_select_menu()
        track_options = ["easy", "medium", "hard"]
        if track_type == "impossible":
            current_track = random.choice(track_options)
            previous_track = current_track
        else:
            current_track = track_type
            previous_track = None
        set_track(current_track)
        cars = [Car(NeuralNetwork()) for _ in range(NUM_CARS)]
    else: # mode == "TEST"
        track_type = track_select_menu()
        track_options = ["easy", "medium", "hard"]
        if track_type == "impossible":
            current_track = random.choice(track_options)
            previous_track = current_track
        else:
            current_track = track_type
            previous_track = None
        set_track(current_track)
        saved_nn = NeuralNetwork.load(saved_ai_path)
        cars = [Car(saved_nn)]

    is_first_frame, generation, best_fitness = True, 1, -float('inf')
    generation_stats, paused, show_all_cars = [], False, True
    save_notification_timer = 0
    total_distance_tested = 0
    
    # <<< CHANGE 1: Use a simulation clock instead of a real-world timer >>>
    simulation_time_elapsed = 0.0

    back_button_rect = pygame.Rect(10, SIM_HEIGHT - 40, 160, 30)
    back_button_color, back_button_hover_color = (220, 220, 220), (200, 200, 200)

    simulation_running = True
    while simulation_running:
        delta_time = min(clock.tick(60) / 1000.0, 0.05)
        
        # <<< CHANGE 2: Increment the simulation clock >>>
        simulation_time_elapsed += delta_time

        if save_notification_timer > 0: save_notification_timer -= delta_time

        for event in pygame.event.get():
            if event.type == pygame.QUIT: pygame.quit(); sys.exit()
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: simulation_running = False
                elif event.key == pygame.K_SPACE: paused = not paused
                elif event.key == pygame.K_a: show_all_cars = not show_all_cars
                elif event.key == pygame.K_s and mode == "TRAIN":
                    best_car = max(cars, key=lambda car: car.fitness) if cars else None
                    if best_car:
                        save_dir = "saved_ais"
                        existing_saves = glob.glob(os.path.join(save_dir, f"AI_{current_track}_Gen-{generation:02d}_*.npz"))
                        next_id = len(existing_saves) + 1
                        filename = os.path.join(save_dir, f"AI_{current_track}_Gen-{generation:02d}_{next_id:02d}.npz")
                        best_car.nn.save(filename)
                        save_notification_timer = 2
                elif event.key == pygame.K_n and mode == "TRAIN":
                     # Fast-forward the simulation clock to end the generation
                     simulation_time_elapsed = GENERATION_DURATION
            if event.type == pygame.MOUSEBUTTONDOWN and back_button_rect.collidepoint(event.pos): simulation_running = False
        if paused: continue

        screen.fill(WHITE); draw_track()
        
        alive_count = 0
        for car in cars:
            if car.alive: car.decide_actions(); car.update(delta_time); alive_count += 1
        
        best_car = max(cars, key=lambda car: car.fitness) if cars else None
        
        if mode == "TEST":
            time_up = simulation_time_elapsed >= GENERATION_DURATION
            if (best_car and not best_car.alive) or time_up:
                total_distance_tested += best_car.distance_traveled if best_car else 0
                simulation_time_elapsed = 0.0 # Reset timer for the next track
                if track_type == "impossible":
                    available_tracks = [t for t in track_options if t != current_track]
                    current_track = random.choice(available_tracks) if available_tracks else random.choice(track_options)
                    previous_track = current_track
                    set_track(current_track)
                else:
                    set_track(current_track)
                saved_nn = best_car.nn if best_car else NeuralNetwork.load(saved_ai_path)
                cars = [Car(saved_nn)]
                best_car = cars[0]

        if show_all_cars:
            for car in cars: draw_car(car, car==best_car)
        elif best_car:
            draw_car(best_car, True)

        if best_car:
            inputs = np.concatenate([best_car.cast_rays(), [best_car.steering_angle/MAX_STEERING_ANGLE, best_car.current_checkpoint/len(checkpoints), best_car.calculate_angle_to_next_checkpoint()]])
            draw_neural_network(NN_PANEL_RECT, best_car.nn, inputs)
            if mode == "TRAIN": draw_performance_graph(GRAPH_RECT, generation_stats)

        mouse_pos = pygame.mouse.get_pos(); hover = back_button_rect.collidepoint(mouse_pos)
        pygame.draw.rect(screen, back_button_hover_color if hover else back_button_color, back_button_rect, border_radius=5)
        pygame.draw.rect(screen, BLACK, back_button_rect, 1, border_radius=5)
        back_text = FONT_SMALL.render("Back to Menu (Esc)", True, BLACK); text_rect = back_text.get_rect(center=back_button_rect.center)
        screen.blit(back_text, text_rect)

        stats = [f"FPS: {clock.get_fps():.1f}"]
        if mode == "TRAIN":
            stats.extend([
                f"Gen: {generation}",
                # <<< CHANGE 3: Display the simulation time >>>
                f"Time: {simulation_time_elapsed:.1f}s / {GENERATION_DURATION}s",
                f"Alive: {alive_count}/{NUM_CARS}",
                f"Best Fit: {max(c.fitness for c in cars):.1f}", "Press 'S' to save"
            ])
        else:
            stats.extend([
                "--- TEST MODE ---",
                # <<< CHANGE 3: Display the simulation time >>>
                f"Time: {simulation_time_elapsed:.1f}s / {GENERATION_DURATION}s",
                f"Total Distance: {total_distance_tested:.0f}"
            ])
        if best_car: stats.extend([f"Laps: {best_car.lap_count}", f"CP: {best_car.current_checkpoint}/{len(checkpoints)}", f"Dist: {best_car.distance_traveled:.0f}"])
        if track_type == "impossible": stats.append(f"Track: {current_track}")
        for i, text in enumerate(stats): screen.blit(FONT_SMALL.render(text, True, BLACK), (10, 10 + i * 24))
        
        if save_notification_timer > 0:
            save_text = FONT_MEDIUM.render("Best AI Saved!", True, BLUE); screen.blit(save_text, (SIM_WIDTH/2 - save_text.get_width()/2, 10))

        # <<< CHANGE 3: Use simulation time for the end condition >>>
        if mode == "TRAIN" and (simulation_time_elapsed >= GENERATION_DURATION or alive_count == 0):
            best_fitness_gen = max(c.fitness for c in cars); avg_fitness_gen = sum(c.fitness for c in cars)/len(cars)
            generation_stats.append({'generation': generation, 'best_fitness': best_fitness_gen, 'avg_fitness': avg_fitness_gen})
            
            if track_type == "impossible":
                available = [t for t in track_options if t != previous_track]; new_track = random.choice(available) if available else random.choice(track_options)
                current_track, previous_track = new_track, new_track
                set_track(current_track)
            else: set_track(current_track)
            
            new_gen = [Car(nn) for nn in elitism(cars, max(3, NUM_CARS//10))]
            top_cars = sorted(cars, key=lambda x:x.fitness, reverse=True)[:5]
            for i, top_car in enumerate(top_cars):
                for _ in range(2 + (2 if i==0 else 0)): new_gen.append(Car(top_car.nn.mutate()))
            while len(new_gen) < NUM_CARS:
                p1, p2 = select_parents(cars), select_parents(cars)
                child_nn = p1.nn.crossover(p2.nn) if random.random()<0.85 else (p1.nn.clone() if random.random()<0.5 else p2.nn.clone())
                new_gen.append(Car(child_nn.mutate(mutation_rate=MUTATION_RATE*(1.0-min(0.5,generation/100)))))
            cars = new_gen; generation += 1; is_first_frame = True
            
            # Reset simulation clock for the new generation
            simulation_time_elapsed = 0.0
            
            pygame.display.flip(); pygame.time.delay(500)

        pygame.display.flip()

if __name__ == "__main__":
    while True:
        mode = main_menu()
        saved_ai_path = None
        if mode == "TEST":
            saved_ai_path = load_ai_menu()
            if saved_ai_path is None: # User pressed escape in load menu
                continue
        run_simulation(mode=mode, saved_ai_path=saved_ai_path)