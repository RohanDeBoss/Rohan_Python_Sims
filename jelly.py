import pygame
import pymunk
import pymunk.pygame_util
import math
import sys
import time

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (214, 73, 54)       # Truck Body
DARK_RED = (179, 58, 45)   # Truck Cab
GREEN = (126, 200, 80)     # Lighter Green (unused here)
BLUE = (70, 172, 217)      # Sky Background
PURPLE = (103, 58, 132)    # Example Obstacle Color
YELLOW = (255, 193, 7)     # Finish Line / Some Obstacles
ORANGE = (255, 87, 34)     # Some Obstacles
DARK_GREEN = (60, 130, 50) # Ground Color

# Physics Constants
GRAVITY = (0, -900) # Pymunk's y-axis is typically upwards
TRUCK_MASS = 50
WHEEL_MASS = 5 # Still unused for physics, only visual
TRUCK_FRICTION = 0.7      # Shape friction property
WHEEL_FRICTION = 1.2      # Lowered for less sticking (unused for physics)
GROUND_FRICTION = 0.9
TRUCK_ELASTICITY = 0.1    # Bounciness
WHEEL_ELASTICITY = 0.2
GROUND_ELASTICITY = 0.3

# Control Forces
FORWARD_FORCE = 70000
BACKWARD_FORCE = 35000
ROTATION_TORQUE = 2000000 # Might need tuning based on feel

# Damping (applied manually in loop)
LINEAR_DAMPING = 0.998
ANGULAR_DAMPING = 0.97

# Collision Types
COLLISION_TYPE_TRUCK = 1
COLLISION_TYPE_WHEEL = 2
COLLISION_TYPE_GROUND = 3
COLLISION_TYPE_OBSTACLE = 4
COLLISION_TYPE_FINISH = 5

# Visual Constants
GROUND_BOTTOM_Y_PYMUNK = -1000 # How far down the filled ground extends in Pymunk Y coords

# --- Game Classes ---

class Truck:
    def __init__(self, space, pos):
        self.space = space
        self.chassis_dims = (70, 40) # Width, Height
        wheel_radius = 18
        wheel_offset_x = 28
        wheel_offset_y = -15 # Relative to chassis center, Y downwards in local Pymunk coords

        # --- Chassis ---
        self.chassis_body = pymunk.Body(TRUCK_MASS, pymunk.moment_for_box(TRUCK_MASS, self.chassis_dims))
        self.chassis_body.position = pos
        self.chassis_shape = pymunk.Poly.create_box(self.chassis_body, self.chassis_dims, radius=1.0)
        self.chassis_shape.friction = TRUCK_FRICTION
        self.chassis_shape.elasticity = TRUCK_ELASTICITY
        self.chassis_shape.collision_type = COLLISION_TYPE_TRUCK
        self.space.add(self.chassis_body, self.chassis_shape)

        # --- Wheels (Visual Only) ---
        self.wheel_radius = wheel_radius
        self.wheel_positions_local = [
            pymunk.Vec2d(-wheel_offset_x, wheel_offset_y),
            pymunk.Vec2d(wheel_offset_x, wheel_offset_y)
        ]
        self.wheel_angle = 0

        self.initial_pos = pos
        self.initial_angle = 0.0

    def reset(self):
        self.chassis_body.position = self.initial_pos
        self.chassis_body.angle = self.initial_angle
        self.chassis_body.velocity = (0, 0)
        self.chassis_body.angular_velocity = 0
        self.wheel_angle = 0

    def apply_force_forward(self):
        direction = pymunk.Vec2d(1, 0).rotated(self.chassis_body.angle)
        # Apply force at the approximate Y-level of the wheel axles for slightly better feel
        force_pos_local = pymunk.Vec2d(0, self.wheel_positions_local[0].y)
        force_pos_world = self.chassis_body.local_to_world(force_pos_local)
        self.chassis_body.apply_force_at_world_point(direction * FORWARD_FORCE, force_pos_world)

    def apply_force_backward(self):
        direction = pymunk.Vec2d(1, 0).rotated(self.chassis_body.angle)
        force_pos_local = pymunk.Vec2d(0, self.wheel_positions_local[0].y)
        force_pos_world = self.chassis_body.local_to_world(force_pos_local)
        self.chassis_body.apply_force_at_world_point(-direction * BACKWARD_FORCE, force_pos_world)
        self.chassis_body.angular_velocity *= 0.95 # Slight braking torque

    def apply_torque_forward(self): # Rotate chassis visually forward (clockwise in Pygame screen)
        self.chassis_body.torque -= ROTATION_TORQUE # Pymunk: positive torque is CCW

    def apply_torque_backward(self): # Rotate chassis visually backward (counter-clockwise in Pygame screen)
        self.chassis_body.torque += ROTATION_TORQUE # Pymunk: negative torque is CW

    def update_visual_wheels(self, dt):
        forward_vec = self.chassis_body.rotation_vector
        velocity_along_forward = self.chassis_body.velocity.dot(forward_vec)
        angular_change = -velocity_along_forward / self.wheel_radius * dt * 1.5
        self.wheel_angle += angular_change

    def draw(self, screen, draw_options, camera_offset):
        # --- Draw Chassis ---
        chassis_points = self.chassis_shape.get_vertices()
        transformed_chassis_points = [self.chassis_body.local_to_world(p) for p in chassis_points]
        screen_chassis_points = [(int(p.x - camera_offset.x), int(SCREEN_HEIGHT - (p.y - camera_offset.y))) for p in transformed_chassis_points]
        if len(screen_chassis_points) >= 3:
            pygame.draw.polygon(screen, RED, screen_chassis_points)
            pygame.draw.polygon(screen, BLACK, screen_chassis_points, 1)

        # --- Draw Visual Cab (Corrected to be on top) ---
        halfW_chassis = self.chassis_dims[0] / 2.0
        halfH_chassis = self.chassis_dims[1] / 2.0 # This is the Y-coordinate of the chassis top surface in local space

        # Cab dimensions and positions relative to chassis center
        # Positive Y is up in Pymunk local space
        cab_rel_back_x = -20                     # How far cab's back extends from chassis center X
        cab_rel_front_x = 15                     # How far cab's front extends from chassis center X
        cab_base_on_chassis_y = halfH_chassis    # Cab sits on top of chassis
        cab_top_front_y_abs = cab_base_on_chassis_y + 25 # Absolute Y for cab's front top
        cab_top_back_y_abs = cab_base_on_chassis_y + 15  # Absolute Y for cab's back top

        cab_local_points = [
            pymunk.Vec2d(cab_rel_back_x, cab_base_on_chassis_y),       # back bottom of cab
            pymunk.Vec2d(cab_rel_front_x, cab_base_on_chassis_y),      # front bottom of cab
            pymunk.Vec2d(cab_rel_front_x, cab_top_front_y_abs),        # front top of cab (taller)
            pymunk.Vec2d(cab_rel_back_x + 5, cab_top_back_y_abs),      # back top of cab (shorter, slanted forward)
        ]

        transformed_cab_points = [self.chassis_body.local_to_world(p) for p in cab_local_points]
        screen_cab_points = [(int(p.x - camera_offset.x), int(SCREEN_HEIGHT - (p.y - camera_offset.y))) for p in transformed_cab_points]
        if len(screen_cab_points) >= 3:
             pygame.draw.polygon(screen, DARK_RED, screen_cab_points)
             pygame.draw.polygon(screen, BLACK, screen_cab_points, 1)

        # --- Draw Wheels ---
        for local_pos in self.wheel_positions_local:
            world_pos = self.chassis_body.local_to_world(local_pos)
            screen_x = int(world_pos.x - camera_offset.x)
            screen_y = int(SCREEN_HEIGHT - (world_pos.y - camera_offset.y))
            pygame.draw.circle(screen, BLACK, (screen_x, screen_y), self.wheel_radius)

            spoke_end_world = world_pos + pymunk.Vec2d(self.wheel_radius * 0.9, 0).rotated(self.wheel_angle + self.chassis_body.angle) # Add chassis angle for spoke
            spoke_end_screen_x = int(spoke_end_world.x - camera_offset.x)
            spoke_end_screen_y = int(SCREEN_HEIGHT - (spoke_end_world.y - camera_offset.y))
            pygame.draw.line(screen, WHITE, (screen_x, screen_y), (spoke_end_screen_x, spoke_end_screen_y), 3)

# --- Helper Functions ---

def create_static_segment(space, p1, p2, thickness=6, friction=GROUND_FRICTION, elasticity=GROUND_ELASTICITY, collision_type=COLLISION_TYPE_GROUND, color=DARK_GREEN):
    body = space.static_body
    shape = pymunk.Segment(body, p1, p2, thickness / 2.0)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    shape.color = color
    space.add(shape)
    return shape

def create_static_box(space, pos, size, angle=0, friction=GROUND_FRICTION, elasticity=GROUND_ELASTICITY, collision_type=COLLISION_TYPE_OBSTACLE, color=ORANGE):
    half_width = size[0] / 2.0
    half_height = size[1] / 2.0
    points = [
        (-half_width, -half_height), (-half_width, half_height),
        ( half_width,  half_height), ( half_width, -half_height)
    ]
    rad_angle = math.radians(angle)
    transformed_points = [pymunk.Vec2d(*p).rotated(rad_angle) + pos for p in points]
    shape = pymunk.Poly(space.static_body, transformed_points)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    shape.color = color
    space.add(shape)
    return shape

def create_dynamic_circle(space, pos, radius, mass, friction=0.6, elasticity=0.6, collision_type=COLLISION_TYPE_OBSTACLE, color=ORANGE):
    body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = friction
    shape.elasticity = elasticity
    shape.collision_type = collision_type
    shape.color = color
    space.add(body, shape)
    return body, shape

def create_finish_line(space, pos, width = 20, height = 60):
     body = space.static_body
     half_width = width / 2.0
     half_height = height / 2.0
     points = [
         (-half_width, -half_height), (-half_width, half_height),
         ( half_width,  half_height), ( half_width, -half_height)
     ]
     shape = pymunk.Poly(body, points, transform=pymunk.Transform(tx=pos.x, ty=pos.y))
     shape.sensor = True
     shape.collision_type = COLLISION_TYPE_FINISH
     shape.color = YELLOW
     space.add(shape)
     return shape

def draw_pymunk_static_shapes(screen, space, camera_offset):
    for shape in space.static_body.shapes:
        color = getattr(shape, 'color', BLUE)

        if isinstance(shape, pymunk.Segment):
            is_ground_segment = (hasattr(shape, 'collision_type') and shape.collision_type == COLLISION_TYPE_GROUND)

            if is_ground_segment:
                p1_world = shape.a
                p2_world = shape.b
                segment_color = getattr(shape, 'color', DARK_GREEN)

                pb1_world = pymunk.Vec2d(p1_world.x, GROUND_BOTTOM_Y_PYMUNK)
                pb2_world = pymunk.Vec2d(p2_world.x, GROUND_BOTTOM_Y_PYMUNK)

                poly_screen_points = [
                    (int(p1_world.x - camera_offset.x), int(SCREEN_HEIGHT - (p1_world.y - camera_offset.y))),
                    (int(p2_world.x - camera_offset.x), int(SCREEN_HEIGHT - (p2_world.y - camera_offset.y))),
                    (int(pb2_world.x - camera_offset.x), int(SCREEN_HEIGHT - (pb2_world.y - camera_offset.y))),
                    (int(pb1_world.x - camera_offset.x), int(SCREEN_HEIGHT - (pb1_world.y - camera_offset.y)))
                ]
                if len(poly_screen_points) >=3:
                    pygame.draw.polygon(screen, segment_color, poly_screen_points)
                    # Draw a black line on top for definition
                    pygame.draw.line(screen, BLACK, poly_screen_points[0], poly_screen_points[1], 1)
            else: # Draw other non-ground segments as before (if any)
                p1 = shape.a
                p2 = shape.b
                radius = max(1, int(shape.radius))
                sp1_x = int(p1.x - camera_offset.x)
                sp1_y = int(SCREEN_HEIGHT - (p1.y - camera_offset.y))
                sp2_x = int(p2.x - camera_offset.x)
                sp2_y = int(SCREEN_HEIGHT - (p2.y - camera_offset.y))
                pygame.draw.line(screen, color, (sp1_x, sp1_y), (sp2_x, sp2_y), radius * 2)

        elif isinstance(shape, pymunk.Poly):
             if shape.sensor and shape.collision_type == COLLISION_TYPE_FINISH:
                  poly_color = (*getattr(shape, 'color', YELLOW)[:3], 128)
                  width = 1
             else:
                  poly_color = color
                  width = 0
             try:
                 tf = shape.get_transform()
                 verts = shape.get_vertices()
                 poly_points_world = [tf * v for v in verts]
                 screen_points = [
                     (int(p.x - camera_offset.x), int(SCREEN_HEIGHT - (p.y - camera_offset.y)))
                     for p in poly_points_world
                 ]
                 if len(screen_points) >= 3:
                     pygame.draw.polygon(screen, poly_color, screen_points, width)
                     if width == 0: # Outline for non-sensor filled polygons
                          pygame.draw.polygon(screen, BLACK, screen_points, 1)
             except Exception: pass

        elif isinstance(shape, pymunk.Circle):
             try:
                 tf = shape.get_transform()
                 pos = tf.translation
                 screen_x = int(pos.x - camera_offset.x)
                 screen_y = int(SCREEN_HEIGHT - (pos.y - camera_offset.y))
                 pygame.draw.circle(screen, color, (screen_x, screen_y), int(shape.radius))
                 pygame.draw.circle(screen, BLACK, (screen_x, screen_y), int(shape.radius), 1)
             except Exception: pass

def draw_dynamic_shapes(screen, space, camera_offset, truck):
     for body in space.bodies:
         if body.body_type == pymunk.Body.DYNAMIC and body != truck.chassis_body:
             for shape in body.shapes:
                  color = getattr(shape, 'color', ORANGE)
                  if isinstance(shape, pymunk.Circle):
                       pos = body.position
                       screen_x = int(pos.x - camera_offset.x)
                       screen_y = int(SCREEN_HEIGHT - (pos.y - camera_offset.y))
                       pygame.draw.circle(screen, color, (screen_x, screen_y), int(shape.radius))
                       pygame.draw.circle(screen, BLACK, (screen_x, screen_y), int(shape.radius), 1)

def clear_level(space, truck):
     bodies_to_remove = [body for body in space.bodies if body.body_type == pymunk.Body.DYNAMIC and body != truck.chassis_body]
     for body in bodies_to_remove:
         shapes_of_body = list(body.shapes)
         for shape in shapes_of_body:
              if shape in space.shapes: space.remove(shape)
         if body in space.bodies: space.remove(body)

     shapes_to_remove = [shape for shape in space.static_body.shapes]
     for shape in shapes_to_remove:
          if shape in space.shapes: space.remove(shape)

def load_level(space, level_index):
    print(f"Loading Level {level_index} geometry...")
    start_pos = pymunk.Vec2d(100, 150)
    finish_pos = pymunk.Vec2d(1000, 150)
    finish_shape = None

    if level_index == 1:
        start_pos = pymunk.Vec2d(150, 150)
        finish_pos = pymunk.Vec2d(2900, 220)
        create_static_segment(space, (-100, 100), (500, 80))
        create_static_segment(space, (500, 80), (700, 130))
        create_static_segment(space, (700, 130), (1200, 110))
        create_static_segment(space, (1200, 110), (1450, 160))
        create_static_segment(space, (1450, 160), (1800, 150))
        create_static_segment(space, (1800, 150), (2100, 100))
        create_static_segment(space, (2100, 100), (2400, 100))
        create_static_segment(space, (2400, 100), (2800, 180))
        create_static_segment(space, (2800, 180), (3100, 180))
        create_static_box(space, (600, 105), (50, 50), color=ORANGE)
        create_static_box(space, (660, 105), (50, 50), color=ORANGE)
        create_static_box(space, (1600, 150 + 40), (80, 80), color=DARK_RED)
        create_static_box(space, (1700, 150 + 40), (80, 80), color=DARK_RED)
        create_static_box(space, (2250, 100 + 25), (150, 50), color=YELLOW) # Adjusted Y for box center
        create_static_box(space, (2550, 100 + 25), (150, 50), color=YELLOW)
    # ... (other levels remain the same, ensure Y for boxes is center)
    elif level_index == 2:
        start_pos = pymunk.Vec2d(100, 150)
        finish_pos = pymunk.Vec2d(1400, 150)
        create_static_segment(space, (-100, 100), (300, 100))
        create_static_box(space, (400, 100 + 50), (20, 100), color=YELLOW) # Center Y is 100 (ground) + 50 (half_height)
        create_static_box(space, (500, 100 + 50), (20, 100), color=ORANGE)
        create_static_box(space, (600, 100 + 50), (20, 100), color=RED)
        create_static_segment(space, (700, 100), (1000, 150))
        create_static_segment(space, (1000, 150), (1200, 150))
        create_static_box(space, (1150, 80 + 10), (100, 20), color=YELLOW) # Center Y for box
        create_static_segment(space, (1300, 100), (1600, 100))

    elif level_index == 3:
        start_pos = pymunk.Vec2d(100, 100)
        finish_pos = pymunk.Vec2d(1800, 100)
        create_static_segment(space, (0, 50), (200, 50))
        create_static_segment(space, (400, 80), (600, 60))
        create_static_segment(space, (750, 120), (950, 100))
        create_static_box(space, (1200, 50), (200, 10), angle=30, color=YELLOW) # Box pos is center
        create_static_box(space, (1200, 50), (200, 10), angle=-30, color=YELLOW)
        create_static_segment(space, (1400, 50), (1900, 50))

    elif level_index == 4:
        start_pos = pymunk.Vec2d(100, 150)
        finish_pos = pymunk.Vec2d(1800, 150)
        create_static_segment(space, (-100, 100), (400, 80))
        create_static_segment(space, (400, 80), (600, 130))
        create_static_segment(space, (700, 120), (1000, 120), color=RED, collision_type=COLLISION_TYPE_OBSTACLE) # Make bridge an obstacle type for different rendering if desired
        create_static_segment(space, (1100, 130), (1400, 100))
        create_static_segment(space, (1400, 100), (1900, 100))
        create_dynamic_circle(space, (1600, 150 + 40), 40, mass=10, color=ORANGE) # Circle Y is center

    elif level_index == 5:
        start_pos = pymunk.Vec2d(100, 100)
        finish_pos = pymunk.Vec2d(2000, 100)
        create_static_segment(space, (0, 50), (250, 50))
        create_static_segment(space, (600, 80), (900, 60))
        create_static_box(space, (800, 100 + 40), (100, 80), color=PURPLE) # Box Y center
        create_static_box(space, (1200, 80 + 10), (150, 20), color=YELLOW)
        create_static_box(space, (1500, 50), (200, 10), angle=45, color=YELLOW)
        create_static_box(space, (1500, 50), (200, 10), angle=-45, color=YELLOW)
        create_static_segment(space, (1700, 50), (2100, 50))
    else: # Placeholder levels
        level_num = level_index
        start_pos = pymunk.Vec2d(100, 150)
        finish_pos = pymunk.Vec2d(1500 + level_num * 100, 150 + (level_num % 4) * 30)
        create_static_segment(space, (0, 100), (finish_pos.x + 200, 100))
        for i in range(level_num % 5 + 1):
            create_static_box(space, (400 + i * 200 + (level_num % 3)*50, 100 + 25), (40, 50), angle=(i*level_num*5)%45, color=( (i*50+level_num*10)%255, (i*30+level_num*20)%255, (i*10+level_num*30)%255) )
        if level_num % 3 == 0:
            create_dynamic_circle(space, (800 + (level_num%2)*100, 200 + 30), 30, mass=5) # Circle Y center

    finish_shape = create_finish_line(space, finish_pos)
    print(f"Level {level_index} geometry loaded. Start: {start_pos}, Finish: {finish_pos}")
    return start_pos, finish_shape


# --- Main Game Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Simplified Jelly Truck (Python)")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)
    message_font = pygame.font.Font(None, 48)
    sub_message_font = pygame.font.Font(None, 28)

    space = pymunk.Space()
    space.gravity = GRAVITY

    current_level = 1
    max_level = 5 # Reduced for testing, can be 20
    truck = Truck(space, pymunk.Vec2d(100, 150))
    finish_shape = None
    level_finished = False
    game_over = False
    level_start_time = time.time()
    finish_message_display_time = 0
    level_time_taken = 0 # Store level time when finished

    level_complete_flag = [False]
    def finish_collision_handler(arbiter, space, data):
        is_truck_chassis = any(shape.collision_type == COLLISION_TYPE_TRUCK for shape in arbiter.shapes)
        if is_truck_chassis and not level_complete_flag[0]:
            print("Level Complete Collision!")
            level_complete_flag[0] = True
        return True
    handler = space.add_collision_handler(COLLISION_TYPE_TRUCK, COLLISION_TYPE_FINISH)
    handler.begin = finish_collision_handler

    camera_offset = pymunk.Vec2d(0, 0)
    camera_smoothing = 0.08

    def setup_new_level(level_idx):
        nonlocal finish_shape, level_start_time, level_finished, game_over, finish_message_display_time
        nonlocal level_complete_flag
        clear_level(space, truck)
        start_pos_visual, new_finish_shape = load_level(space, level_idx)
        finish_shape = new_finish_shape
        # Pymunk Y is upwards, Pygame Y is downwards. Visual Y assumes bottom of screen is 0.
        # For Pymunk, a higher Y means higher up.
        # If start_pos_visual.y is 150 (meaning 150px from bottom of screen visually),
        # this doesn't directly map to Pymunk Y unless screen origin is considered.
        # The levels seem to define Pymunk Y directly. Let's stick to that.
        # The original code did: start_pos_pymunk = pymunk.Vec2d(start_pos_visual.x, SCREEN_HEIGHT - start_pos_visual.y)
        # This implies start_pos_visual.y was screen-based.
        # However, load_level returns pymunk.Vec2d directly, which should be pymunk coordinates.
        truck.initial_pos = start_pos_visual # Use direct Pymunk coordinates from load_level
        truck.reset()
        level_start_time = time.time()
        level_complete_flag[0] = False
        level_finished = False
        game_over = False
        finish_message_display_time = 0

    setup_new_level(current_level)

    running = True
    while running:
        dt = clock.tick(FPS) / 1000.0
        dt = min(dt, 0.05) # Cap dt

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r:
                    print("Resetting Level...")
                    if game_over: current_level = 1 # If game over, R restarts from level 1
                    setup_new_level(current_level)
                if event.key == pygame.K_n and level_finished and not game_over:
                     current_level += 1
                     if current_level > max_level:
                         game_over = True
                         print("Game Finished!")
                     else:
                         print("Loading Next Level...")
                         setup_new_level(current_level)

        keys = pygame.key.get_pressed()
        if not level_finished and not game_over:
            if keys[pygame.K_UP]: truck.apply_force_forward()
            if keys[pygame.K_DOWN]: truck.apply_force_backward()
            if keys[pygame.K_LEFT]: truck.apply_torque_backward()
            if keys[pygame.K_RIGHT]: truck.apply_torque_forward()

        truck.chassis_body.velocity *= LINEAR_DAMPING
        truck.chassis_body.angular_velocity *= ANGULAR_DAMPING
        truck.update_visual_wheels(dt)
        space.step(dt)

        if level_complete_flag[0] and not level_finished:
            level_finished = True
            level_time_taken = time.time() - level_start_time
            finish_message_display_time = time.time()
            print(f"Internal: Level {current_level} Finished. Time: {level_time_taken:.2f}")

        # --- Camera Update (Locked to truck center, with smoothing) ---
        target_camera_x = truck.chassis_body.position.x - SCREEN_WIDTH / 2
        target_camera_y = truck.chassis_body.position.y - SCREEN_HEIGHT / 2
        camera_offset = pymunk.Vec2d(
            camera_offset.x + (target_camera_x - camera_offset.x) * camera_smoothing,
            camera_offset.y + (target_camera_y - camera_offset.y) * camera_smoothing
        )

        screen.fill(BLUE)
        draw_pymunk_static_shapes(screen, space, camera_offset)
        draw_dynamic_shapes(screen, space, camera_offset, truck)
        truck.draw(screen, None, camera_offset)

        if not game_over:
             current_run_time = time.time() - level_start_time
             if level_finished: # Show stored time once level is done
                 timer_text = font.render(f"Time: {level_time_taken:.2f}", True, WHITE)
             else:
                 timer_text = font.render(f"Time: {current_run_time:.2f}", True, WHITE)
             screen.blit(timer_text, (10, 10))

        level_text = font.render(f"Level: {current_level}", True, WHITE)
        screen.blit(level_text, (SCREEN_WIDTH - level_text.get_width() - 10, 10))

        if level_finished and finish_message_display_time > 0:
            finish_message = message_font.render(f"Level {current_level} Complete!", True, YELLOW)
            finish_rect = finish_message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 30))
            screen.blit(finish_message, finish_rect)

            time_message = small_font.render(f"Time: {level_time_taken:.2f}s", True, WHITE)
            time_rect = time_message.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 10))
            screen.blit(time_message, time_rect)

            if current_level < max_level:
                next_prompt = sub_message_font.render("Press N for Next Level or R to Retry", True, WHITE)
                next_rect = next_prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
                screen.blit(next_prompt, next_rect)
            else:
                 end_prompt = sub_message_font.render("You finished all levels! Press R or ESC", True, WHITE)
                 end_rect = end_prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 50))
                 screen.blit(end_prompt, end_rect)
                 game_over = True # Set game_over flag here once all levels are done

        if game_over and not (level_finished and current_level == max_level) : # Show general game over if not from completing last level
             game_over_text = message_font.render("GAME FINISHED!", True, YELLOW)
             go_rect = game_over_text.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 - 20))
             screen.blit(game_over_text, go_rect)
             final_prompt = sub_message_font.render("Press R to Restart Level 1 or ESC to Exit", True, WHITE)
             final_rect = final_prompt.get_rect(center=(SCREEN_WIDTH // 2, SCREEN_HEIGHT // 2 + 30))
             screen.blit(final_prompt, final_rect)

        pygame.display.flip()

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()