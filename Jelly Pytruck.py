import pygame
import pymunk
# No need for pymunk.pygame_util if not using its draw options
import math
import sys
import time
import random

# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

# Colors - Defined clearly
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (214, 73, 54)         # Truck Body
DARK_RED = (179, 58, 45)    # Truck Cab
BASE_SKY_BLUE = (135, 206, 235) # Main background sky
PURPLE = (103, 58, 132)     # Obstacle Color
YELLOW = (255, 193, 7)      # Finish Line / Some Obstacles
ORANGE = (255, 87, 34)      # Some Obstacles
DARK_GREEN = (60, 130, 50)  # Ground Color
DEBUG_MAGENTA = (255, 0, 255) # For debugging missing colors
GRAY = (128, 128, 128)      # Wheel Color
WINDOW_BLUE = (100, 140, 200) # Truck Window Color

# Background Colors
SKY_HORIZON_BLUE = (100, 180, 220) # Slightly darker blue near horizon
HILL_GREEN_FAR = (50, 120, 40)     # Darker, more distant hills
HILL_GREEN_NEAR = (80, 160, 70)    # Lighter, closer hills
CLOUD_WHITE = (240, 240, 250)      # Slightly off-white clouds

# Physics Constants
GRAVITY = (0, -900)
CHASSIS_MASS = 30
WHEEL_MASS = 1.5
CHASSIS_FRICTION = 0.6
WHEEL_FRICTION = 1.5 # High friction for grip
GROUND_FRICTION = 0.9
CHASSIS_ELASTICITY = 0.25 # A bit bouncy
WHEEL_ELASTICITY = 0.2 # A bit bouncy
GROUND_ELASTICITY = 0.2
OBSTACLE_FRICTION = 0.5 # Less grippy obstacles
OBSTACLE_ELASTICITY = 0.5 # Moderately bouncy obstacles

# Control Forces / Motor Parameters
MOTOR_TARGET_RATE_FORWARD = 27 # Adjust for speed feel
MOTOR_TARGET_RATE_BACKWARD = -27
MOTOR_MAX_FORCE = 200000 # Adjust for acceleration feel
BRAKE_MAX_FORCE = MOTOR_MAX_FORCE * 0.9

ROTATION_TORQUE = 1300000 # Torque for initiating flips

# --- Add this constant for easy tweaking at the top ---
MAX_ANGULAR_VELOCITY_CHASSIS = 10  # Maximum allowed chassis spin velocity (radians/sec)

# NEW: Angular Damping for Chassis
ANGULAR_DRAG_COEFFICIENT = 5000  # Increase for much stronger drag (try 2.0, 3.0, etc.)
ANGULAR_DRAG_POWER = 1          # Quadratic drag is more "real" for air resistance

# Collision Types
COLLISION_TYPE_CHASSIS = 1
COLLISION_TYPE_WHEEL = 2
COLLISION_TYPE_GROUND = 3
COLLISION_TYPE_OBSTACLE = 4
COLLISION_TYPE_FINISH = 5

GROUND_BOTTOM_Y_PYMUNK = -1000 # For drawing filled ground

# Background Parallax Scroll Speeds
BG_DISTANT_SCROLL_X = 0.05
BG_DISTANT_SCROLL_Y = 0.01
BG_MID_SCROLL_X = 0.2
BG_MID_SCROLL_Y = 0.03
BG_NEAR_SCROLL_X = 0.4
BG_NEAR_SCROLL_Y = 0.08

# Helper alias for screen height in drawing code
SH = SCREEN_HEIGHT
SW = SCREEN_WIDTH # Added Screen Width alias

class Truck:
    def __init__(self, space, pos):
        self.space = space
        self.chassis_dims = (70, 30)
        self.wheel_radius = 18

        # --- Chassis ---
        moment = pymunk.moment_for_box(CHASSIS_MASS, self.chassis_dims)
        self.chassis_body = pymunk.Body(CHASSIS_MASS, moment)
        self.chassis_body.position = pos
        # Increased radius for rounder chassis, helps with self-righting
        self.chassis_shape = pymunk.Poly.create_box(self.chassis_body, self.chassis_dims, radius=4.0)
        self.chassis_shape.friction = CHASSIS_FRICTION
        self.chassis_shape.elasticity = CHASSIS_ELASTICITY
        self.chassis_shape.collision_type = COLLISION_TYPE_CHASSIS
        self.space.add(self.chassis_body, self.chassis_shape)
        # REMOVED Limit max angular velocity: self.chassis_body.angular_velocity_limit = MAX_ANGULAR_VELOCITY_CHASSIS

        # --- Wheels ---
        wheel_offset_x = self.chassis_dims[0] * 0.4
        chassis_half_height = self.chassis_dims[1] / 2
        wheel_axle_y_offset_from_center = -chassis_half_height - 2

        self.wheel_positions_local = [
            pymunk.Vec2d(-wheel_offset_x, wheel_axle_y_offset_from_center), # Rear
            pymunk.Vec2d(wheel_offset_x, wheel_axle_y_offset_from_center)   # Front
        ]
        self.wheel_bodies = []
        self.wheel_shapes = []
        self.wheel_motors = []
        self.wheel_joints = [] # Pivot joints

        for i, local_axle_pos in enumerate(self.wheel_positions_local):
            # Wheel Body
            wheel_moment = pymunk.moment_for_circle(WHEEL_MASS, 0, self.wheel_radius)
            wheel_body = pymunk.Body(WHEEL_MASS, wheel_moment)
            wheel_body.position = self.chassis_body.local_to_world(local_axle_pos)

            # Wheel Shape
            wheel_shape = pymunk.Circle(wheel_body, self.wheel_radius)
            wheel_shape.friction = WHEEL_FRICTION
            wheel_shape.elasticity = WHEEL_ELASTICITY
            wheel_shape.collision_type = COLLISION_TYPE_WHEEL
            wheel_shape.color = GRAY
            self.space.add(wheel_body, wheel_shape)
            self.wheel_bodies.append(wheel_body)
            self.wheel_shapes.append(wheel_shape)

            # Pivot Joint
            pivot = pymunk.PivotJoint(self.chassis_body, wheel_body, local_axle_pos, (0,0))
            pivot.collide_bodies = False # Prevent chassis and wheels colliding
            self.space.add(pivot)
            self.wheel_joints.append(pivot)

            # Motor
            motor = pymunk.SimpleMotor(self.chassis_body, wheel_body, 0) # Initial rate 0
            motor.max_force = 0 # Motor initially off
            self.space.add(motor)
            self.wheel_motors.append(motor)

        self.initial_pos = pymunk.Vec2d(pos.x, pos.y) # Store initial position
        self.initial_angle = 0.0
        self.reset() # Call reset to finalize state

    def reset(self):
        self.chassis_body.position = self.initial_pos
        self.chassis_body.angle = self.initial_angle
        self.chassis_body.velocity = (0, 0)
        self.chassis_body.angular_velocity = 0

        for i, wheel_body in enumerate(self.wheel_bodies):
            local_axle_pos = self.wheel_positions_local[i]
            wheel_body.position = self.chassis_body.local_to_world(local_axle_pos)
            wheel_body.angle = self.chassis_body.angle
            wheel_body.velocity = (0,0)
            wheel_body.angular_velocity = 0
            motor = self.wheel_motors[i]
            motor.rate = 0
            motor.max_force = 0

    def drive_forward(self):
        for motor in self.wheel_motors:
            motor.rate = MOTOR_TARGET_RATE_FORWARD
            motor.max_force = MOTOR_MAX_FORCE

    def drive_backward(self):
        for motor in self.wheel_motors:
            motor.rate = MOTOR_TARGET_RATE_BACKWARD
            motor.max_force = MOTOR_MAX_FORCE

    # Renamed for clarity based on visual effect
    def apply_torque_nose_down(self):
        self.chassis_body.torque -= ROTATION_TORQUE # Negative torque for CW screen

    def apply_torque_nose_up(self):
        self.chassis_body.torque += ROTATION_TORQUE # Positive torque for CCW screen

    def release_throttle_and_brake(self):
        for motor in self.wheel_motors:
            motor.rate = 0
            motor.max_force = BRAKE_MAX_FORCE / 5 # Gentle braking effect

    def draw(self, screen, camera_offset):
        co = camera_offset # Use local alias

        # --- Draw Chassis ---
        chassis_verts_world = [self.chassis_body.local_to_world(v) for v in self.chassis_shape.get_vertices()]
        screen_pts_chassis = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in chassis_verts_world]
        if len(screen_pts_chassis) >= 3:
            pygame.draw.polygon(screen, RED, screen_pts_chassis)
            # Outline drawn AFTER cab and window below

        # --- Calculate Cab and Window Vertices (Relative to Chassis Center) ---
        chassis_half_w = self.chassis_dims[0] / 2.0
        chassis_half_h = self.chassis_dims[1] / 2.0
        cab_back_x = 5; cab_front_x = chassis_half_w - 5
        cab_bottom_y = chassis_half_h; cab_top_front_y = cab_bottom_y + 25
        cab_top_back_y = cab_bottom_y + 18
        cab_local_pts = [
            pymunk.Vec2d(cab_back_x, cab_bottom_y), pymunk.Vec2d(cab_front_x, cab_bottom_y),
            pymunk.Vec2d(cab_front_x, cab_top_front_y), pymunk.Vec2d(cab_back_x, cab_top_back_y)
        ]
        window_inset_h = 4; window_inset_v = 4
        win_back_x = cab_back_x + window_inset_h; win_front_x = cab_front_x - window_inset_h
        win_bottom_y = cab_bottom_y + window_inset_v; win_top_y = cab_top_back_y - window_inset_v
        window_local_pts = [
            pymunk.Vec2d(win_back_x, win_bottom_y), pymunk.Vec2d(win_front_x, win_bottom_y),
            pymunk.Vec2d(win_front_x, win_top_y), pymunk.Vec2d(win_back_x, win_top_y)
        ]

        # --- Transform and Draw Cab ---
        cab_verts_world = [self.chassis_body.local_to_world(p) for p in cab_local_pts]
        screen_pts_cab = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in cab_verts_world]
        if len(screen_pts_cab) >= 3:
             pygame.draw.polygon(screen, DARK_RED, screen_pts_cab)

        # --- Transform and Draw Window ---
        window_verts_world = [self.chassis_body.local_to_world(p) for p in window_local_pts]
        screen_pts_window = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in window_verts_world]
        if len(screen_pts_window) >= 3:
            pygame.draw.polygon(screen, WINDOW_BLUE, screen_pts_window) # Fill window

        # --- Draw Outlines (AFTER fills) ---
        if len(screen_pts_chassis) >= 3:
             pygame.draw.polygon(screen, BLACK, screen_pts_chassis, 1) # Chassis outline
        if len(screen_pts_cab) >= 3:
             pygame.draw.polygon(screen, BLACK, screen_pts_cab, 1) # Cab outline

        # --- Draw Wheels ---
        for i, wheel_body in enumerate(self.wheel_bodies):
            wheel_pos_world = wheel_body.position
            screen_x = int(wheel_pos_world.x - co.x)
            screen_y = int(SH - (wheel_pos_world.y - co.y))
            wheel_color = getattr(self.wheel_shapes[i], 'color', GRAY)
            pygame.draw.circle(screen, wheel_color, (screen_x, screen_y), self.wheel_radius)
            pygame.draw.circle(screen, BLACK, (screen_x, screen_y), self.wheel_radius, 1)

            spoke_end_world = wheel_body.local_to_world((self.wheel_radius * 0.9, 0))
            spoke_end_screen_x = int(spoke_end_world.x - co.x)
            spoke_end_screen_y = int(SH - (spoke_end_world.y - co.y))
            pygame.draw.line(screen, WHITE, (screen_x, screen_y), (spoke_end_screen_x, spoke_end_screen_y), 2)


# --- Helper Functions ---
def create_static_segment(space, p1, p2, **kwargs):
    defaults = {'thickness': 6, 'friction': GROUND_FRICTION, 'elasticity': GROUND_ELASTICITY,
                'collision_type': COLLISION_TYPE_GROUND, 'color': DARK_GREEN}
    props = {**defaults, **kwargs}
    shape = pymunk.Segment(space.static_body, p1, p2, props['thickness'] / 2.0)
    shape.friction = props['friction']
    shape.elasticity = props['elasticity']
    shape.collision_type = props['collision_type']
    shape.color = props['color']
    space.add(shape)
    return shape

def create_box(space, pos, size, mass=None, angle=0, **kwargs):
    defaults = {'friction': OBSTACLE_FRICTION, 'elasticity': OBSTACLE_ELASTICITY,
                'collision_type': COLLISION_TYPE_OBSTACLE, 'color': ORANGE, 'radius': 0.5}
    props = {**defaults, **kwargs}

    if mass is None: # Static box
        body = space.static_body
        half_w, half_h = size[0]/2, size[1]/2
        local_pts = [(-half_w,-half_h),(-half_w,half_h),(half_w,half_h),(half_w,-half_h)]
        transform = pymunk.Transform.identity().translated(pos.x,pos.y).rotated(math.radians(angle))
        shape = pymunk.Poly(body, local_pts, transform=transform, radius=props['radius'])
    else: # Dynamic box
        body = pymunk.Body(mass, pymunk.moment_for_box(mass, size))
        body.position = pos
        body.angle = math.radians(angle)
        shape = pymunk.Poly.create_box(body, size, radius=props['radius'])
        space.add(body)

    shape.friction = props['friction']
    shape.elasticity = props['elasticity']
    shape.collision_type = props['collision_type']
    shape.color = props['color']
    space.add(shape)
    return (body, shape) if mass is not None else (None, shape)

def create_dynamic_circle(space, pos, radius, mass, **kwargs):
    defaults = {'friction': OBSTACLE_FRICTION, 'elasticity': OBSTACLE_ELASTICITY,
                'collision_type': COLLISION_TYPE_OBSTACLE, 'color': ORANGE}
    props = {**defaults, **kwargs}
    body = pymunk.Body(mass, pymunk.moment_for_circle(mass, 0, radius))
    body.position = pos
    shape = pymunk.Circle(body, radius)
    shape.friction = props['friction']
    shape.elasticity = props['elasticity']
    shape.collision_type = props['collision_type']
    shape.color = props['color']
    space.add(body, shape)
    return body, shape

def create_finish_line(space, pos, width=20, height=80): # Made slightly taller
    half_w, half_h = width/2, height/2
    pts = [(-half_w,-half_h),(-half_w,half_h),(half_w,half_h),(half_w,-half_h)]
    xfm = pymunk.Transform(tx=pos.x,ty=pos.y)
    shp = pymunk.Poly(space.static_body, pts, transform=xfm)
    shp.sensor = True # Make it a sensor so truck passes through
    shp.collision_type = COLLISION_TYPE_FINISH
    shp.color = YELLOW
    space.add(shp)
    return shp

def draw_pymunk_static_shapes(screen, space, camera_offset):
    co = camera_offset
    for shape in space.static_body.shapes:
        color = getattr(shape, 'color', DEBUG_MAGENTA)
        if isinstance(shape, pymunk.Segment):
            p1_world, p2_world = shape.a, shape.b
            if hasattr(shape, 'collision_type') and shape.collision_type == COLLISION_TYPE_GROUND:
                segment_color = getattr(shape, 'color', DARK_GREEN)
                pb1 = pymunk.Vec2d(p1_world.x, GROUND_BOTTOM_Y_PYMUNK)
                pb2 = pymunk.Vec2d(p2_world.x, GROUND_BOTTOM_Y_PYMUNK)
                poly_points = [
                    (int(p1_world.x - co.x), int(SH - (p1_world.y - co.y))),
                    (int(p2_world.x - co.x), int(SH - (p2_world.y - co.y))),
                    (int(pb2.x - co.x),    int(SH - (pb2.y - co.y))),
                    (int(pb1.x - co.x),    int(SH - (pb1.y - co.y)))
                ]
                # Prevent drawing invalid polygons if points collapse
                if len(set(poly_points)) >= 3: # Need at least 3 unique points
                    try:
                        pygame.draw.polygon(screen, segment_color[:3], poly_points)
                    except ValueError: # Handle potential error if points are collinear
                        print(f"Warning: Collinear points for ground polygon: {poly_points}")
                # Draw top line segment
                pygame.draw.line(screen, BLACK, poly_points[0], poly_points[1], 1)
            else: # Other static segments
                radius = max(1, int(shape.radius))
                sp1 = (int(p1_world.x - co.x), int(SH - (p1_world.y - co.y)))
                sp2 = (int(p2_world.x - co.x), int(SH - (p2_world.y - co.y)))
                # Check if points are the same to avoid zero-length line error
                if sp1 != sp2:
                    try:
                        pygame.draw.line(screen, color[:3], sp1, sp2, int(radius * 2))
                    except ValueError:
                         print(f"Warning: Invalid line parameters: {sp1}, {sp2}, {int(radius*2)}")

        elif isinstance(shape, pymunk.Poly):
            poly_color, poly_width = color, 0
            if shape.sensor and shape.collision_type == COLLISION_TYPE_FINISH:
                poly_color = YELLOW
            verts_world = [shape.body.local_to_world(v) for v in shape.get_vertices()]
            screen_points = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in verts_world]
            if len(screen_points) >= 3:
                try:
                    pygame.draw.polygon(screen, poly_color[:3], screen_points, poly_width)
                except ValueError:
                    print(f"Warning: Collinear points for static polygon: {screen_points}")

                if poly_width == 0 and not (shape.sensor and shape.collision_type == COLLISION_TYPE_FINISH):
                     # Add outline only for non-sensor filled polygons
                    try:
                        pygame.draw.polygon(screen, BLACK, screen_points, 1)
                    except ValueError:
                        pass # Ignore outline error if fill worked

def draw_dynamic_shapes(screen, space, camera_offset, truck):
    co = camera_offset
    ignore_bodies = [truck.chassis_body] + truck.wheel_bodies
    for body in space.bodies:
        if body.body_type == pymunk.Body.DYNAMIC and body not in ignore_bodies:
            for shape in body.shapes:
                color = getattr(shape, 'color', ORANGE)
                if isinstance(shape, pymunk.Circle):
                    pos_world = body.local_to_world(shape.offset)
                    screen_x = int(pos_world.x - co.x)
                    screen_y = int(SH - (pos_world.y - co.y))
                    radius = int(shape.radius)
                    if radius > 0: # Ensure radius is positive
                        pygame.draw.circle(screen, color[:3], (screen_x, screen_y), radius)
                        pygame.draw.circle(screen, BLACK, (screen_x, screen_y), radius, 1)
                elif isinstance(shape, pymunk.Poly):
                    verts_world = [body.local_to_world(v) for v in shape.get_vertices()]
                    screen_points = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in verts_world]
                    if len(screen_points) >= 3:
                        try:
                            pygame.draw.polygon(screen, color[:3], screen_points, 0)
                            pygame.draw.polygon(screen, BLACK, screen_points, 1)
                        except ValueError:
                             print(f"Warning: Collinear points for dynamic polygon: {screen_points}")


def clear_level(space, truck):
    # Remove dynamic bodies NOT part of the truck
    bodies_to_remove = [b for b in space.bodies if b.body_type == pymunk.Body.DYNAMIC and \
                        b != truck.chassis_body and b not in truck.wheel_bodies]
    for body in bodies_to_remove:
        # Important: Remove constraints FIRST
        for constraint in list(body.constraints):
             if constraint in space.constraints: space.remove(constraint)
        for shape in list(body.shapes):
            if shape in space.shapes: space.remove(shape)
        if body in space.bodies: space.remove(body)

    # Remove static shapes
    for shape in list(space.static_body.shapes):
        if shape in space.shapes: space.remove(shape)

# --- UPDATED LEVEL DESIGNS ---
def load_level(space, level_index):
    print(f"Loading Level {level_index}...")
    base_y = 150 # Base ground height
    start_pos = pymunk.Vec2d(150, base_y + 50) # Default start
    finish_pos = pymunk.Vec2d(1000, base_y + 40) # Default finish

    def cgp(points, **kwargs): # create_ground_path alias
        for i in range(len(points) - 1): create_static_segment(space, points[i], points[i+1], **kwargs)

    if level_index == 1: # Gentle Intro + First Dynamic Boxes
        theme_color = ORANGE
        finish_pos = pymunk.Vec2d(2800, base_y + 50)
        gp = [(-200, base_y), (600, base_y), (800, base_y + 20), (1100, base_y + 30), (1400, base_y),
              (1700, base_y - 10), (2000, base_y - 10), (2300, base_y + 40), (2600, base_y + 40),
              (finish_pos.x + 200, base_y + 40)]
        cgp(gp)
        create_box(space, pymunk.Vec2d(950, base_y + 55), (50, 50), mass=5, color=theme_color)
        create_box(space, pymunk.Vec2d(1800, base_y + 15), (100, 30), mass=8, color=theme_color)
        create_box(space, pymunk.Vec2d(2450, base_y + 65), (60, 40), mass=6, color=theme_color)

    elif level_index == 2: # Steeper Hills, Small Jump, More Dynamics
        theme_color = YELLOW
        start_pos = pymunk.Vec2d(100, base_y + 50)
        finish_pos = pymunk.Vec2d(3000, base_y + 100)
        gp = [(-100, base_y), (300, base_y), (600, base_y + 80), (900, base_y + 80),
              (1100, base_y + 20), (1400, base_y + 30), (1600, base_y + 100), # Ramp up
              # Gap from 1650 to 1800 for jump
              (1800, base_y + 100), # Landing ramp
              (2100, base_y + 40), (2400, base_y + 40), (2700, base_y + 120),
              (finish_pos.x + 100, base_y + 120)]
        cgp(gp)
        create_box(space, pymunk.Vec2d(750, base_y + 105), (40, 40), mass=4, color=theme_color)
        create_dynamic_circle(space, pymunk.Vec2d(1250, base_y + 80), 25, mass=5, color=theme_color)
        # Add more obstacles after jump
        create_box(space, pymunk.Vec2d(1950, base_y + 65), (70, 30), mass=7, color=theme_color)
        create_box(space, pymunk.Vec2d(2500, base_y + 150), (50, 50), mass=6, color=theme_color) # Higher obstacle
        create_dynamic_circle(space, pymunk.Vec2d(2200, base_y + 80), 20, mass=4, color=RED)

    elif level_index == 3: # Tricky Terrain, Static Obstacles Mix
        theme_color = PURPLE
        start_pos = pymunk.Vec2d(100, base_y + 30)
        finish_pos = pymunk.Vec2d(3400, base_y)
        gp = [(-100, base_y), (200, base_y), (350, base_y - 40), (500, base_y - 40), # Drop
              (650, base_y + 60), (800, base_y + 60), (900, base_y + 10), # Hump
              (1200, base_y + 10), (1400, base_y - 30), (1700, base_y - 30),
              (1900, base_y + 80), (2100, base_y + 80), # Platform
              (2200, base_y), (2500, base_y), (2700, base_y - 50), # Drop
              (3100, base_y - 50), (3300, base_y + 10),
              (finish_pos.x + 100, base_y + 10)]
        cgp(gp)
        # Static obstacles forcing path
        create_box(space, pymunk.Vec2d(425, base_y + 10), (100, 20), mass=None, angle=-15, color=theme_color) # Static Ramp
        create_box(space, pymunk.Vec2d(1050, base_y + 40), (20, 80), mass=None, color=theme_color) # Static Wall
        # Dynamic obstacles in tricky spots
        create_box(space, pymunk.Vec2d(1800, base_y + 5), (80, 80), mass=15, color=theme_color)
        create_box(space, pymunk.Vec2d(2350, base_y + 50), (100, 100), mass=30, color=theme_color) # Large dynamic block
        create_dynamic_circle(space, pymunk.Vec2d(2850, base_y + 30), 30, mass=8, color=theme_color)

    elif level_index == 4: # Bridge/Jump Focus, Rolling Hazards
        theme_color = RED
        start_pos = pymunk.Vec2d(100, base_y + 80)
        finish_pos = pymunk.Vec2d(3600, base_y + 50)
        gp = [(-100, base_y + 40), (400, base_y + 40), (600, base_y + 100), # Ramp up
              # Gap 1: 650 to 950
              (950, base_y + 100), (1200, base_y + 60), # Downslope
              (1500, base_y + 60), (1700, base_y + 120), # Ramp up
              # Gap 2: 1800 to 2100
              (2100, base_y + 120), (2400, base_y + 80), # Landing
              (2700, base_y + 80), (2900, base_y + 100), (3200, base_y + 100),
              (finish_pos.x + 100, base_y + 100)]
        cgp(gp)
        # Dynamic Bridge Planks
        create_box(space, pymunk.Vec2d(800, base_y + 95), (320, 15), mass=25, color=theme_color)
        create_box(space, pymunk.Vec2d(1950, base_y + 115), (320, 15), mass=25, color=theme_color)
        # Rolling Circles from hills
        create_dynamic_circle(space, pymunk.Vec2d(1650, base_y + 180), 35, mass=10, color=ORANGE)
        create_dynamic_circle(space, pymunk.Vec2d(1750, base_y + 220), 25, mass=7, color=ORANGE)
        create_dynamic_circle(space, pymunk.Vec2d(2850, base_y + 150), 40, mass=12, color=ORANGE)
        create_box(space, pymunk.Vec2d(3100, base_y + 125), (60, 60), mass=12, color=theme_color)

    elif level_index == 5: # Longer, Varied Climb/Descent - FIXED HEIGHT
        theme_color = DARK_GREEN
        base_y = 200 # Start this level higher off the bottom
        start_pos = pymunk.Vec2d(100, base_y + 50) # Ensure start is above base
        finish_pos = pymunk.Vec2d(4000, base_y + 100)
        path = [(-100, base_y)] # Start path at the higher base
        cx = -100; cy = base_y
        # Generate path segments ensuring Y doesn't drop too low relative to new base_y
        sections = [(500, 30), (400, -50), (600, 80), (500, -20),
                    (700, 100), (600, 20), (800, -60), (finish_pos.x - cx, 50)]
        min_y = base_y - 80 # Set a minimum Y floor for path generation
        for dx, dy_change in sections:
            cx += dx
            # Add randomness but clamp to min_y
            target_y = cy + dy_change + random.randint(-20, 20)
            cy = max(min_y, target_y)
            path.append((cx, cy))
        cgp(path, color=DARK_GREEN)

        # Add obstacles relative to the path height
        create_box(space,pymunk.Vec2d(800,path[1][1]+60),(120,120),mass=40,color=PURPLE)
        create_dynamic_circle(space,pymunk.Vec2d(1500,path[2][1]+80),50,mass=25,color=ORANGE)
        create_box(space,pymunk.Vec2d(2200,path[4][1]+40),(200,30),mass=20,angle=-10,color=PURPLE)
        create_box(space,pymunk.Vec2d(2800,path[5][1]+75),(80,150),mass=None,color=DARK_RED) # Tall static
        create_dynamic_circle(space,pymunk.Vec2d(3400,path[6][1]+70),40,mass=15,color=ORANGE)

    else: # Placeholder remains simple
        ln=level_index-5; start_pos=pymunk.Vec2d(100,base_y+50)
        finish_pos=pymunk.Vec2d(1500+ln*200,base_y+(ln%3)*40)
        path=[(-100,base_y)]; cx=-100
        for _ in range(4+ln%3):cx+=random.randint(300,500);path.append((cx,base_y+random.randint(-30,50)))
        path.append((finish_pos.x+100,path[-1][1])); cgp(path)
        for i in range(2+ln%4):
            ox,oy=400+i*(300+ln*20),base_y+25+random.randint(0,50)
            create_box(space,pymunk.Vec2d(ox,oy),(40,50),mass=5+i,color=ORANGE)

    finish_shape = create_finish_line(space, finish_pos)
    print(f"Level {level_index} loaded. Start: {start_pos}, Finish: {finish_pos}")
    return start_pos, finish_shape


# --- Main Game Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Jelly Truck Adventures - Tuned")
    clock = pygame.time.Clock()
    font = pygame.font.Font(None, 36)
    small_font = pygame.font.Font(None, 24)

    space = pymunk.Space()
    space.gravity = GRAVITY

    current_level = 1
    max_level = 5
    truck = Truck(space, pymunk.Vec2d(150, 250)) # Initial pos before level load

    level_finished = False; game_over = False
    level_start_time = time.time(); level_time_taken = 0
    level_complete_flag = [False]

    bg_elements = [
        {'y_on_screen_top':0,'height_on_screen':int(SH*.6),'color':SKY_HORIZON_BLUE,'scroll_x':BG_DISTANT_SCROLL_X,'scroll_y':.01},
        {'y_on_screen_top':int(SH*.5),'height_on_screen':int(SH*.3),'color':HILL_GREEN_FAR,'scroll_x':BG_MID_SCROLL_X,'scroll_y':BG_MID_SCROLL_Y},
        {'y_on_screen_top':int(SH*.7),'height_on_screen':int(SH*.3),'color':HILL_GREEN_NEAR,'scroll_x':BG_NEAR_SCROLL_X,'scroll_y':BG_NEAR_SCROLL_Y}]
    clouds = []
    for _ in range(10): clouds.append([random.randint(-SW,SW*2),random.randint(int(SH*.1),int(SH*.4)),random.randint(20,40),random.randint(15,35),random.randint(10,30)])

    def finish_collision_handler(arbiter, space, data):
        is_chassis = any(hasattr(s,'collision_type') and s.collision_type==COLLISION_TYPE_CHASSIS for s in arbiter.shapes)
        if is_chassis and not level_complete_flag[0]: print("Level Complete!"); level_complete_flag[0]=True
        return True
    handler_chassis = space.add_collision_handler(COLLISION_TYPE_CHASSIS, COLLISION_TYPE_FINISH); handler_chassis.begin=finish_collision_handler

    camera_offset = pymunk.Vec2d(0,0); camera_smoothing = 0.08

    def setup_new_level(level_idx):
        nonlocal level_start_time,level_finished,game_over,level_complete_flag,level_time_taken,truck,camera_offset
        clear_level(space,truck)
        start_pos_pymunk,_ = load_level(space,level_idx)
        truck.initial_pos = start_pos_pymunk; truck.reset()
        camera_offset = pymunk.Vec2d(truck.chassis_body.position.x-SW/2,truck.chassis_body.position.y-SH/2)
        level_start_time=time.time(); level_complete_flag[0]=False; level_finished=False; level_time_taken=0

    setup_new_level(current_level)
    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.033)

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r:
                    if game_over and not (level_finished and current_level==max_level): clvl,gover=1,False
                    setup_new_level(current_level)
                if event.key == pygame.K_n and level_finished and not (game_over and current_level==max_level):
                    current_level+=1;
                    if current_level>max_level: game_over=True
                    else: setup_new_level(current_level)
            if event.type == pygame.KEYUP:
                if not level_finished and not game_over:
                    if event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s): truck.release_throttle_and_brake()

        keys = pygame.key.get_pressed()
        if not level_finished and not game_over:
            drive_action = False
            if keys[pygame.K_UP] or keys[pygame.K_w]: truck.drive_forward(); drive_action=True
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: truck.drive_backward(); drive_action=True
            if not drive_action: truck.release_throttle_and_brake()

            # Apply user torque
            if keys[pygame.K_LEFT] or keys[pygame.K_a]: truck.apply_torque_nose_up()
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: truck.apply_torque_nose_down()

            # --- Apply strong angular drag for chassis ---
            av = truck.chassis_body.angular_velocity
            if abs(av) > 0.01:
                drag_torque = -ANGULAR_DRAG_COEFFICIENT * (abs(av) ** ANGULAR_DRAG_POWER) * math.copysign(1, av)
                truck.chassis_body.torque += drag_torque

            # --- Clamp max angular velocity for chassis (hard limit) ---
            max_spin = MAX_ANGULAR_VELOCITY_CHASSIS
            if av > max_spin:
                truck.chassis_body.angular_velocity = max_spin
            elif av < -max_spin:
                truck.chassis_body.angular_velocity = -max_spin

        # --- Physics Step ---
        space.step(dt)

        # --- Game Logic Update ---
        if level_complete_flag[0] and not level_finished:
            level_finished = True; level_time_taken = time.time() - level_start_time
            print(f"Internal: Level {current_level} Finished. Time: {level_time_taken:.2f}")
            if current_level == max_level: game_over = True

        # --- Camera Update ---
        target_cam_center_x = truck.chassis_body.position.x
        target_cam_center_y = truck.chassis_body.position.y
        target_offset_x = target_cam_center_x - SW / 2
        target_offset_y = target_cam_center_y - SH / 2
        new_cam_offset_x = camera_offset.x + (target_offset_x - camera_offset.x) * camera_smoothing
        new_cam_offset_y = camera_offset.y + (target_offset_y - camera_offset.y) * camera_smoothing
        camera_offset = pymunk.Vec2d(new_cam_offset_x, new_cam_offset_y)

        # --- Drawing ---
        screen.fill(BASE_SKY_BLUE) # Base sky

        # Draw Background Layers
        for layer in bg_elements:
            layer_x = - (camera_offset.x * layer['scroll_x']) % SW
            adj_y = layer['y_on_screen_top'] - (camera_offset.y * layer['scroll_y'])
            pygame.draw.rect(screen, layer['color'], (layer_x, adj_y, SW, layer['height_on_screen']))
            pygame.draw.rect(screen, layer['color'], (layer_x + SW, adj_y, SW, layer['height_on_screen']))
            pygame.draw.rect(screen, layer['color'], (layer_x - SW, adj_y, SW, layer['height_on_screen']))

        # Draw Clouds
        cloud_sx, cloud_sy = BG_MID_SCROLL_X, BG_MID_SCROLL_Y
        for c_data in clouds:
            base_x, base_y, r1, r2, r3 = c_data
            cwx_app = base_x - camera_offset.x * cloud_sx
            sww = SW * 1.5; csx = (cwx_app + SW / 2) % sww - SW * 0.25
            csy = base_y - camera_offset.y * cloud_sy
            max_r = max(r1,r2,r3)
            if -max_r < csx < SW + max_r:
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx),int(csy)),r1)
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx+r1*.7),int(csy+5)),r2)
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx-r1*.6),int(csy+3)),r3)

        # Draw Game World
        draw_pymunk_static_shapes(screen, space, camera_offset)
        draw_dynamic_shapes(screen, space, camera_offset, truck)
        truck.draw(screen, camera_offset)

        # --- UI Drawing ---
        timer_str = f"Time: {level_time_taken:.2f}" if level_finished else f"Time: {(time.time() - level_start_time):.2f}"
        timer_surf = font.render(timer_str, True, WHITE)
        screen.blit(timer_surf, (10, 10))

        level_str = f"Level: {current_level}"
        level_surf = font.render(level_str, True, WHITE)
        screen.blit(level_surf, (SW - level_surf.get_width() - 10, 10))

        if level_finished:
            msg_text = "Level Complete!" if current_level < max_level else "All Levels Complete!"
            prompt_text = "N: Next, R: Retry" if current_level < max_level else "R: Restart, ESC: Exit"
            msg_surf = font.render(msg_text, True, YELLOW)
            prompt_surf = small_font.render(prompt_text, True, WHITE)
            msg_rect = msg_surf.get_rect(center=(SW // 2, SH // 2 - 20))
            prompt_rect = prompt_surf.get_rect(center=(SW // 2, SH // 2 + 20))
            screen.blit(msg_surf, msg_rect)
            screen.blit(prompt_surf, prompt_rect)

        pygame.display.flip() # Update the full screen

    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()