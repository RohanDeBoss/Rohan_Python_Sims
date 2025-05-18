import pygame
import pymunk
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
GROUND_FRICTION = 0.95
CHASSIS_ELASTICITY = 0.25 # A bit bouncy
WHEEL_ELASTICITY = 0.2 # A bit bouncy
GROUND_ELASTICITY = 0.2
OBSTACLE_FRICTION = 0.5 # Less grippy obstacles
OBSTACLE_ELASTICITY = 0.5 # Moderately bouncy obstacles

# Control Forces / Motor Parameters
MOTOR_TARGET_RATE_FORWARD = 27
MOTOR_TARGET_RATE_BACKWARD = -27
MOTOR_MAX_FORCE = 200000
BRAKE_MAX_FORCE = MOTOR_MAX_FORCE * 0.9
ROTATION_TORQUE = 1300000

MAX_ANGULAR_VELOCITY_CHASSIS = 10
ANGULAR_DRAG_COEFFICIENT = 5000
ANGULAR_DRAG_POWER = 1

# Collision Types
COLLISION_TYPE_CHASSIS = 1
COLLISION_TYPE_WHEEL = 2
COLLISION_TYPE_GROUND = 3
COLLISION_TYPE_OBSTACLE = 4
COLLISION_TYPE_FINISH = 5

GROUND_BOTTOM_Y_PYMUNK = -1000
BG_DISTANT_SCROLL_X = 0.05; BG_DISTANT_SCROLL_Y = 0.01
BG_MID_SCROLL_X = 0.2; BG_MID_SCROLL_Y = 0.03
BG_NEAR_SCROLL_X = 0.4; BG_NEAR_SCROLL_Y = 0.08
SH = SCREEN_HEIGHT; SW = SCREEN_WIDTH

class Truck:
    def __init__(self, space, pos):
        self.space = space
        self.chassis_dims = (70, 30)
        self.wheel_radius = 18

        moment = pymunk.moment_for_box(CHASSIS_MASS, self.chassis_dims)
        self.chassis_body = pymunk.Body(CHASSIS_MASS, moment)
        self.chassis_body.position = pos
        self.chassis_shape = pymunk.Poly.create_box(self.chassis_body, self.chassis_dims, radius=4.0)
        self.chassis_shape.friction = CHASSIS_FRICTION
        self.chassis_shape.elasticity = CHASSIS_ELASTICITY
        self.chassis_shape.collision_type = COLLISION_TYPE_CHASSIS
        self.space.add(self.chassis_body, self.chassis_shape)

        wheel_offset_x = self.chassis_dims[0] * 0.4
        chassis_half_height = self.chassis_dims[1] / 2
        wheel_axle_y_offset_from_center = -chassis_half_height - 2

        self.wheel_positions_local = [
            pymunk.Vec2d(-wheel_offset_x, wheel_axle_y_offset_from_center),
            pymunk.Vec2d(wheel_offset_x, wheel_axle_y_offset_from_center)
        ]
        self.wheel_bodies = []
        self.wheel_shapes = []
        self.wheel_motors = []
        self.wheel_joints = []

        for i, local_axle_pos in enumerate(self.wheel_positions_local):
            wheel_moment = pymunk.moment_for_circle(WHEEL_MASS, 0, self.wheel_radius)
            wheel_body = pymunk.Body(WHEEL_MASS, wheel_moment)
            wheel_body.position = self.chassis_body.local_to_world(local_axle_pos)
            wheel_shape = pymunk.Circle(wheel_body, self.wheel_radius)
            wheel_shape.friction = WHEEL_FRICTION; wheel_shape.elasticity = WHEEL_ELASTICITY
            wheel_shape.collision_type = COLLISION_TYPE_WHEEL; wheel_shape.color = GRAY
            self.space.add(wheel_body, wheel_shape)
            self.wheel_bodies.append(wheel_body); self.wheel_shapes.append(wheel_shape)
            pivot = pymunk.PivotJoint(self.chassis_body, wheel_body, local_axle_pos, (0,0))
            pivot.collide_bodies = False; self.space.add(pivot); self.wheel_joints.append(pivot)
            motor = pymunk.SimpleMotor(self.chassis_body, wheel_body, 0)
            motor.max_force = 0; self.space.add(motor); self.wheel_motors.append(motor)

        self.initial_pos = pymunk.Vec2d(pos.x, pos.y)
        self.initial_angle = 0.0
        self.reset()

    def reset(self):
        self.chassis_body.position = self.initial_pos; self.chassis_body.angle = self.initial_angle
        self.chassis_body.velocity = (0,0); self.chassis_body.angular_velocity = 0
        for i, wb in enumerate(self.wheel_bodies):
            wb.position = self.chassis_body.local_to_world(self.wheel_positions_local[i])
            wb.angle = self.chassis_body.angle; wb.velocity = (0,0); wb.angular_velocity = 0
            self.wheel_motors[i].rate = 0; self.wheel_motors[i].max_force = 0

    def drive_forward(self):
        for m in self.wheel_motors: m.rate = MOTOR_TARGET_RATE_FORWARD; m.max_force = MOTOR_MAX_FORCE
    def drive_backward(self):
        for m in self.wheel_motors: m.rate = MOTOR_TARGET_RATE_BACKWARD; m.max_force = MOTOR_MAX_FORCE
    def apply_torque_nose_down(self): self.chassis_body.torque -= ROTATION_TORQUE
    def apply_torque_nose_up(self): self.chassis_body.torque += ROTATION_TORQUE
    def release_throttle_and_brake(self):
        for m in self.wheel_motors: m.rate = 0; m.max_force = BRAKE_MAX_FORCE / 5

    def draw(self, screen, camera_offset):
        co = camera_offset
        ch_verts_w = [self.chassis_body.local_to_world(v) for v in self.chassis_shape.get_vertices()]
        scr_pts_ch = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in ch_verts_w]
        if len(scr_pts_ch) >= 3: pygame.draw.polygon(screen, RED, scr_pts_ch)

        ch_hw, ch_hh = self.chassis_dims[0]/2.0, self.chassis_dims[1]/2.0
        cb_bx, cb_fx = 5, ch_hw - 5
        cb_by, cb_tfy, cb_tby = ch_hh, ch_hh + 25, ch_hh + 18
        cab_loc_pts = [pymunk.Vec2d(cb_bx, cb_by), pymunk.Vec2d(cb_fx, cb_by), pymunk.Vec2d(cb_fx, cb_tfy), pymunk.Vec2d(cb_bx, cb_tby)]
        win_ih, win_iv = 4, 4
        wn_bx, wn_fx = cb_bx + win_ih, cb_fx - win_ih
        wn_by, wn_ty = cb_by + win_iv, cb_tby - win_iv
        win_loc_pts = [pymunk.Vec2d(wn_bx, wn_by), pymunk.Vec2d(wn_fx, wn_by), pymunk.Vec2d(wn_fx, wn_ty), pymunk.Vec2d(wn_bx, wn_ty)]

        cab_verts_w = [self.chassis_body.local_to_world(p) for p in cab_loc_pts]
        scr_pts_cab = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in cab_verts_w]
        if len(scr_pts_cab) >= 3: pygame.draw.polygon(screen, DARK_RED, scr_pts_cab)

        win_verts_w = [self.chassis_body.local_to_world(p) for p in win_loc_pts]
        scr_pts_win = [(int(p.x - co.x), int(SH - (p.y - co.y))) for p in win_verts_w]
        if len(scr_pts_win) >= 3: pygame.draw.polygon(screen, WINDOW_BLUE, scr_pts_win)

        if len(scr_pts_ch) >= 3: pygame.draw.polygon(screen, BLACK, scr_pts_ch, 1)
        if len(scr_pts_cab) >= 3: pygame.draw.polygon(screen, BLACK, scr_pts_cab, 1)

        for i, wb in enumerate(self.wheel_bodies):
            wp_w = wb.position
            sx, sy = int(wp_w.x - co.x), int(SH - (wp_w.y - co.y))
            wc = getattr(self.wheel_shapes[i], 'color', GRAY)
            pygame.draw.circle(screen, wc, (sx, sy), self.wheel_radius)
            pygame.draw.circle(screen, BLACK, (sx, sy), self.wheel_radius, 1)
            sp_end_w = wb.local_to_world((self.wheel_radius*0.9,0))
            sp_esx, sp_esy = int(sp_end_w.x-co.x), int(SH-(sp_end_w.y-co.y))
            pygame.draw.line(screen, WHITE, (sx,sy), (sp_esx,sp_esy), 2)

def create_static_segment(space, p1, p2, **kwargs):
    defaults = {'thickness':6,'friction':GROUND_FRICTION,'elasticity':GROUND_ELASTICITY,'collision_type':COLLISION_TYPE_GROUND,'color':DARK_GREEN}
    props = {**defaults, **kwargs}; shp = pymunk.Segment(space.static_body,p1,p2,props['thickness']/2.0)
    shp.friction=props['friction']; shp.elasticity=props['elasticity']; shp.collision_type=props['collision_type']; shp.color=props['color']
    space.add(shp); return shp

def create_box(space, pos, size, mass=None, angle=0, **kwargs):
    defaults = {'friction':OBSTACLE_FRICTION,'elasticity':OBSTACLE_ELASTICITY,'collision_type':COLLISION_TYPE_OBSTACLE,'color':ORANGE,'radius':0.5}
    props = {**defaults, **kwargs}
    if mass is None:
        body = space.static_body; hw,hh = size[0]/2,size[1]/2; lpts = [(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)]
        xfm = pymunk.Transform.identity().translated(pos.x,pos.y).rotated(math.radians(angle))
        shp = pymunk.Poly(body, lpts, transform=xfm, radius=props['radius'])
    else:
        body = pymunk.Body(mass, pymunk.moment_for_box(mass,size)); body.position = pos; body.angle = math.radians(angle)
        shp = pymunk.Poly.create_box(body, size, radius=props['radius']); space.add(body)
    shp.friction=props['friction']; shp.elasticity=props['elasticity']; shp.collision_type=props['collision_type']; shp.color=props['color']
    space.add(shp); return (body,shp) if mass is not None else (None,shp)

def create_dynamic_circle(space, pos, radius, mass, **kwargs):
    defaults = {'friction':OBSTACLE_FRICTION,'elasticity':OBSTACLE_ELASTICITY,'collision_type':COLLISION_TYPE_OBSTACLE,'color':ORANGE}
    props = {**defaults, **kwargs}; body = pymunk.Body(mass, pymunk.moment_for_circle(mass,0,radius)); body.position = pos
    shp = pymunk.Circle(body, radius)
    shp.friction=props['friction']; shp.elasticity=props['elasticity']; shp.collision_type=props['collision_type']; shp.color=props['color']
    space.add(body,shp); return body,shp

def create_finish_line(space, pos, width=20, height=80):
    hw,hh = width/2,height/2; pts = [(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)]; xfm = pymunk.Transform(tx=pos.x,ty=pos.y)
    shp = pymunk.Poly(space.static_body, pts, transform=xfm); shp.sensor = True
    shp.collision_type = COLLISION_TYPE_FINISH; shp.color = YELLOW; space.add(shp); return shp

def draw_pymunk_static_shapes(screen, space, camera_offset):
    co = camera_offset
    for shape in space.static_body.shapes:
        color = getattr(shape,'color',DEBUG_MAGENTA)
        if isinstance(shape, pymunk.Segment):
            p1w,p2w = shape.a,shape.b
            if hasattr(shape,'collision_type') and shape.collision_type == COLLISION_TYPE_GROUND:
                seg_c = getattr(shape,'color',DARK_GREEN); pb1 = pymunk.Vec2d(p1w.x,GROUND_BOTTOM_Y_PYMUNK); pb2 = pymunk.Vec2d(p2w.x,GROUND_BOTTOM_Y_PYMUNK)
                poly_pts = [(int(p1w.x-co.x),int(SH-(p1w.y-co.y))),(int(p2w.x-co.x),int(SH-(p2w.y-co.y))),(int(pb2.x-co.x),int(SH-(pb2.y-co.y))),(int(pb1.x-co.x),int(SH-(pb1.y-co.y)))]
                if len(set(poly_pts)) >= 3:
                    try: pygame.draw.polygon(screen,seg_c[:3],poly_pts)
                    except ValueError: print(f"Warn: CollGroundPoly: {poly_pts}")
                pygame.draw.line(screen,BLACK,poly_pts[0],poly_pts[1],1)
            else:
                rad = max(1,int(shape.radius)); sp1=(int(p1w.x-co.x),int(SH-(p1w.y-co.y))); sp2=(int(p2w.x-co.x),int(SH-(p2w.y-co.y)))
                if sp1 != sp2:
                    try: pygame.draw.line(screen,color[:3],sp1,sp2,int(rad*2))
                    except ValueError: print(f"Warn: InvLine: {sp1},{sp2},{int(rad*2)}")
        elif isinstance(shape, pymunk.Poly):
            poly_c,poly_w = color,0
            if shape.sensor and shape.collision_type == COLLISION_TYPE_FINISH: poly_c = YELLOW
            verts_w = [shape.body.local_to_world(v) for v in shape.get_vertices()]
            scr_pts = [(int(p.x-co.x),int(SH-(p.y-co.y))) for p in verts_w]
            if len(scr_pts)>=3:
                try: pygame.draw.polygon(screen,poly_c[:3],scr_pts,poly_w)
                except ValueError: print(f"Warn: CollStaticPoly: {scr_pts}")
                if poly_w==0 and not (shape.sensor and shape.collision_type==COLLISION_TYPE_FINISH):
                    try: pygame.draw.polygon(screen,BLACK,scr_pts,1)
                    except ValueError: pass

def draw_dynamic_shapes(screen, space, camera_offset, truck):
    co = camera_offset; ignore_b = [truck.chassis_body] + truck.wheel_bodies
    for body in space.bodies:
        if body.body_type == pymunk.Body.DYNAMIC and body not in ignore_b:
            for shp in body.shapes:
                color = getattr(shp,'color',ORANGE)
                if isinstance(shp, pymunk.Circle):
                    pos_w = body.local_to_world(shp.offset); sx,sy = int(pos_w.x-co.x),int(SH-(pos_w.y-co.y)); rad = int(shp.radius)
                    if rad > 0: pygame.draw.circle(screen,color[:3],(sx,sy),rad); pygame.draw.circle(screen,BLACK,(sx,sy),rad,1)
                elif isinstance(shp, pymunk.Poly):
                    verts_w = [body.local_to_world(v) for v in shp.get_vertices()]; scr_pts = [(int(p.x-co.x),int(SH-(p.y-co.y))) for p in verts_w]
                    if len(scr_pts)>=3:
                        try: pygame.draw.polygon(screen,color[:3],scr_pts,0); pygame.draw.polygon(screen,BLACK,scr_pts,1)
                        except ValueError: print(f"Warn: CollDynPoly: {scr_pts}")

def clear_level(space, truck):
    bodies_to_remove = [b for b in space.bodies if b.body_type==pymunk.Body.DYNAMIC and b!=truck.chassis_body and b not in truck.wheel_bodies]
    for body in bodies_to_remove:
        for c in list(body.constraints):
             if c in space.constraints: space.remove(c)
        for s in list(body.shapes):
            if s in space.shapes: space.remove(s)
        if body in space.bodies: space.remove(body)
    for s in list(space.static_body.shapes):
        if s in space.shapes: space.remove(s)

# --- REVAMPED LEVEL DESIGNS ---
def load_level(space, level_index):
    print(f"Loading Level {level_index}...")
    base_y = 150 # Base ground height for most levels
    start_pos = pymunk.Vec2d(150, base_y + 50) # Default start, truck center Y is base_y + 50
    # Rule: Ground at finish_pos.x will be finish_pos.y - 50
    # Finish line center is finish_pos.y. Height 80 means bottom is finish_pos.y - 40.
    # Truck fits if ground is below finish_pos.y - 40. (finish_pos.y - 50) works.
    ground_at_finish_offset = -50

    def cgp(points, **kwargs):
        for i in range(len(points) - 1): create_static_segment(space, points[i], points[i+1], **kwargs)

    if level_index == 1: # Gentle Intro + First Dynamic Boxes
        theme_color = ORANGE
        finish_pos = pymunk.Vec2d(2800, base_y + 60) # Finish line center Y = 210
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset # Y = 160
        gp = [(-200, base_y), (600, base_y), (800, base_y + 20), (1100, base_y + 30), (1400, base_y),
              (1700, base_y - 10), (2000, base_y - 10), (2300, base_y + 40),
              (finish_pos.x, ground_y_at_finish), # Ground at finish_x
              (finish_pos.x + 200, ground_y_at_finish)] # Extend flat
        cgp(gp)
        create_box(space, pymunk.Vec2d(950, base_y + 55), (50, 50), mass=5, color=theme_color)
        create_box(space, pymunk.Vec2d(1800, base_y + 15), (100, 30), mass=8, color=theme_color)
        create_box(space, pymunk.Vec2d(2450, base_y + 65), (60, 40), mass=6, color=theme_color)

    elif level_index == 2: # Steeper Hills, Small Jump, More Dynamics
        theme_color = YELLOW
        start_pos = pymunk.Vec2d(100, base_y + 50)
        finish_pos = pymunk.Vec2d(3000, base_y + 120) # Finish line center Y = 270
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset # Y = 220
        gp = [(-100, base_y), (300, base_y), (600, base_y + 80), (900, base_y + 80),
              (1100, base_y + 20), (1400, base_y + 30), (1600, base_y + 100), # Ramp up for jump
              # Gap from 1600 to 1750 (150 units wide)
              (1750, base_y + 100), # Landing ramp
              (2100, base_y + 40), (2400, base_y + 40), (2700, base_y + 120),
              (finish_pos.x, ground_y_at_finish),
              (finish_pos.x + 100, ground_y_at_finish)]
        cgp(gp)
        create_box(space, pymunk.Vec2d(750, base_y + 105), (40, 40), mass=4, color=theme_color)
        create_dynamic_circle(space, pymunk.Vec2d(1250, base_y + 80), 25, mass=5, color=theme_color)
        create_box(space, pymunk.Vec2d(1950, base_y + 65), (70, 30), mass=7, color=theme_color)
        create_box(space, pymunk.Vec2d(2500, base_y + 150), (50, 50), mass=6, color=theme_color)
        create_dynamic_circle(space, pymunk.Vec2d(2200, base_y + 80), 20, mass=4, color=RED)

    elif level_index == 3: # Tricky Terrain, Static Obstacles Mix
        theme_color = PURPLE
        start_pos = pymunk.Vec2d(100, base_y + 30)
        finish_pos = pymunk.Vec2d(3400, base_y + 20) # Finish line center Y = 170
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset # Y = 120
        gp = [(-100, base_y), (200, base_y), (350, base_y - 40), (500, base_y - 40), # Drop
              (650, base_y + 60), (800, base_y + 60), (900, base_y + 10), # Hump
              (1200, base_y + 10), (1400, base_y - 30), (1700, base_y - 30),
              (1900, base_y + 70), (2100, base_y + 80), # Platform
              (2200, base_y), (2500, base_y), (2700, base_y - 50), # Drop
              (3100, base_y - 50),
              (finish_pos.x, ground_y_at_finish),
              (finish_pos.x + 100, ground_y_at_finish)]
        cgp(gp)
        # Adjusted static ramp to potentially help out of the first dip
        create_box(space, pymunk.Vec2d(425, base_y - 40 + 10), (100, 20), mass=None, angle=-20, color=theme_color)
        create_box(space, pymunk.Vec2d(1050, base_y + 40), (20, 80), mass=None, color=theme_color) # Static Wall
        create_box(space, pymunk.Vec2d(1800, base_y + 5), (80, 80), mass=15, color=theme_color)
        create_box(space, pymunk.Vec2d(2350, base_y + 50), (100, 100), mass=20, color=theme_color) # Lighter large block
        create_dynamic_circle(space, pymunk.Vec2d(2850, base_y + 30), 30, mass=8, color=theme_color)

    elif level_index == 4: # Bridge/Jump Focus, Rolling Hazards
        theme_color = RED
        start_pos = pymunk.Vec2d(100, base_y + 80)
        finish_pos = pymunk.Vec2d(3600, base_y + 70) # Finish Y = 220
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset # Y = 170

        # Adjusted ground points for 200-unit gaps
        gp = [(-100, base_y + 40), (600, base_y + 100), # Ramp up, end of platform for Gap 1 (X=600)
              # Gap 1: X from 600 to 800 (200 units wide)
              (800, base_y + 100), (1200, base_y + 60), # Downslope after Gap 1
              (1500, base_y + 60), (1700, base_y + 120), # Ramp up, end of platform for Gap 2 (X=1700)
              # Gap 2: X from 1700 to 1900 (200 units wide)
              (1900, base_y + 120), (2400, base_y + 80), # Landing after Gap 2
              (2700, base_y + 80), (2900, base_y + 100), (3200, base_y + 100),
              (finish_pos.x, ground_y_at_finish),
              (finish_pos.x + 100, ground_y_at_finish)]
        cgp(gp)

        # Dynamic Bridge Planks (adjusted size and position for 200-unit gaps)
        create_box(space, pymunk.Vec2d(700, base_y + 95), (220, 15), mass=15, color=theme_color) # For gap 600-800
        create_box(space, pymunk.Vec2d(1800, base_y + 115), (220, 15), mass=15, color=theme_color) # For gap 1700-1900

        create_dynamic_circle(space, pymunk.Vec2d(1650, base_y + 180), 35, mass=10, color=ORANGE)
        create_dynamic_circle(space, pymunk.Vec2d(1750, base_y + 220), 25, mass=7, color=ORANGE)
        create_dynamic_circle(space, pymunk.Vec2d(2850, base_y + 150), 40, mass=12, color=ORANGE)
        create_box(space, pymunk.Vec2d(3100, base_y + 125), (60, 60), mass=12, color=theme_color)

    elif level_index == 5: # Longer, Varied Climb/Descent
        theme_color = DARK_GREEN
        lvl5_base_y = 200 # Start this level higher off the bottom
        start_pos = pymunk.Vec2d(100, lvl5_base_y + 50)
        finish_pos = pymunk.Vec2d(4000, lvl5_base_y + 100) # Finish Y = 300
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset # Y = 250

        path = [(-100, lvl5_base_y)]
        cx, cy = -100, lvl5_base_y
        min_y_allowed = lvl5_base_y - 80
        # Sections describe (dx, dy_change_from_current_cy)
        sections_data = [(500, 30), (400, -50), (600, 80), (500, -20),
                         (700, 100), (600, 20), (800, -60)]

        for dx, dy_change in sections_data:
            cx += dx
            target_y = cy + dy_change + random.randint(-20, 20)
            cy = max(min_y_allowed, target_y)
            path.append((cx, cy))

        # Ensure the path reaches finish_pos.x with the correct ground height
        path.append((finish_pos.x, ground_y_at_finish))
        path.append((finish_pos.x + 200, ground_y_at_finish)) # Extend flat
        cgp(path, color=DARK_GREEN)

        # Add obstacles relative to path heights or absolute if appropriate
        create_box(space,pymunk.Vec2d(800, path[2][1]+30),(120,60),mass=20,color=PURPLE) # path[2] is after first two segments
        create_dynamic_circle(space,pymunk.Vec2d(1500, path[3][1]+40),50,mass=15,color=ORANGE)
        create_box(space,pymunk.Vec2d(2200, path[5][1]+20),(200,30),mass=10,angle=-10,color=PURPLE)
        create_box(space,pymunk.Vec2d(2800, lvl5_base_y + 75),(80,150),mass=None,color=DARK_RED) # Tall static relative to level base
        create_dynamic_circle(space,pymunk.Vec2d(3400, path[7][1]+35),40,mass=10,color=ORANGE)


    else: # Placeholder: Simple flat ground with random finish height
        ln = level_index - 5
        start_pos = pymunk.Vec2d(100, base_y + 50)
        finish_y_val = base_y + (ln % 3) * 40 + 40 # Ensure finish is reasonably high
        finish_pos = pymunk.Vec2d(1500 + ln * 200, finish_y_val)
        ground_y_at_finish = finish_pos.y + ground_at_finish_offset

        path = [(-100, base_y)]
        current_x = -100
        for _ in range(3 + ln % 2): # Fewer, longer segments
            current_x += random.randint(400, 600)
            path.append((current_x, base_y + random.randint(-20, 30)))
        path.append((finish_pos.x, ground_y_at_finish))
        path.append((finish_pos.x + 100, ground_y_at_finish))
        cgp(path)
        for i in range(1 + ln % 3):
            ox = 400 + i * (400 + ln * 30)
            oy = base_y + 25 + random.randint(0, 40) # Obstacles on the main ground
            create_box(space, pymunk.Vec2d(ox, oy), (40,50), mass=5+i*2, color=ORANGE)

    finish_shape = create_finish_line(space, finish_pos)
    print(f"Level {level_index} loaded. Start: {start_pos}, Finish: {finish_pos.x, finish_pos.y}, Ground @ Finish X: {ground_y_at_finish if 'ground_y_at_finish' in locals() else 'N/A'}")
    return start_pos, finish_shape


# --- Main Game Function (Largely unchanged from your provided one) ---
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
    max_level = 5 # You can increase this as you add/test more levels
    truck = Truck(space, pymunk.Vec2d(150, 250))

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
        start_pos_pymunk,_ = load_level(space,level_idx) # Finish_shape is not used here
        truck.initial_pos = start_pos_pymunk; truck.reset()
        # Initialize camera to truck's start position
        camera_offset = pymunk.Vec2d(truck.chassis_body.position.x-SW/2,truck.chassis_body.position.y-SH/2)
        level_start_time=time.time(); level_complete_flag[0]=False; level_finished=False; level_time_taken=0; game_over = False


    setup_new_level(current_level)
    running = True
    while running:
        dt = min(clock.tick(FPS) / 1000.0, 0.033) # Cap dt for stability

        for event in pygame.event.get():
            if event.type == pygame.QUIT: running = False
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                if event.key == pygame.K_r:
                    # If game is over (all levels done or some other future condition), R restarts from level 1
                    if game_over and not (level_finished and current_level == max_level) :
                         current_level = 1
                         game_over = False # Reset game_over state
                    setup_new_level(current_level)

                if event.key == pygame.K_n and level_finished and not (current_level == max_level and game_over):
                    current_level+=1
                    if current_level > max_level:
                        game_over=True # All defined levels complete
                        # Don't setup new level here, let the game_over message show
                    else:
                        setup_new_level(current_level)

            if event.type == pygame.KEYUP:
                if not level_finished and not game_over:
                    if event.key in (pygame.K_UP, pygame.K_w, pygame.K_DOWN, pygame.K_s):
                        truck.release_throttle_and_brake()

        keys = pygame.key.get_pressed()
        if not level_finished and not game_over:
            drive_action = False
            if keys[pygame.K_UP] or keys[pygame.K_w]: truck.drive_forward(); drive_action=True
            if keys[pygame.K_DOWN] or keys[pygame.K_s]: truck.drive_backward(); drive_action=True
            if not drive_action: truck.release_throttle_and_brake()

            if keys[pygame.K_LEFT] or keys[pygame.K_a]: truck.apply_torque_nose_up()
            if keys[pygame.K_RIGHT] or keys[pygame.K_d]: truck.apply_torque_nose_down()

            av = truck.chassis_body.angular_velocity
            if abs(av) > 0.01:
                drag_torque = -ANGULAR_DRAG_COEFFICIENT * (abs(av)**ANGULAR_DRAG_POWER) * math.copysign(1,av)
                truck.chassis_body.torque += drag_torque
            max_spin = MAX_ANGULAR_VELOCITY_CHASSIS
            if av > max_spin: truck.chassis_body.angular_velocity = max_spin
            elif av < -max_spin: truck.chassis_body.angular_velocity = -max_spin

        space.step(dt)

        if level_complete_flag[0] and not level_finished:
            level_finished = True; level_time_taken = time.time() - level_start_time
            print(f"Internal: Level {current_level} Finished. Time: {level_time_taken:.2f}")
            if current_level == max_level: game_over = True # Set game_over if last level is finished

                # --- Camera Update ---
        target_cam_center_x = truck.chassis_body.position.x
        target_cam_center_y = truck.chassis_body.position.y
        target_offset_x = target_cam_center_x - SW / 2
        target_offset_y = target_cam_center_y - SH / 2

        # Calculate the new camera offset components
        new_camera_x = camera_offset.x + (target_offset_x - camera_offset.x) * camera_smoothing
        new_camera_y = camera_offset.y + (target_offset_y - camera_offset.y) * camera_smoothing

        # Assign a new Vec2d object to camera_offset
        camera_offset = pymunk.Vec2d(new_camera_x, new_camera_y)

        screen.fill(BASE_SKY_BLUE)
        for layer in bg_elements:
            lx = -(camera_offset.x*layer['scroll_x'])%SW; ay = layer['y_on_screen_top']-(camera_offset.y*layer['scroll_y'])
            pygame.draw.rect(screen,layer['color'],(lx,ay,SW,layer['height_on_screen']))
            pygame.draw.rect(screen,layer['color'],(lx+SW,ay,SW,layer['height_on_screen']))
            pygame.draw.rect(screen,layer['color'],(lx-SW,ay,SW,layer['height_on_screen']))
        csx_sc,csy_sc = BG_MID_SCROLL_X,BG_MID_SCROLL_Y
        for c_dat in clouds:
            bx,by,r1,r2,r3 = c_dat; cwx = bx-camera_offset.x*csx_sc; sww=SW*1.5; csx=(cwx+SW/2)%sww-SW*.25
            csy = by-camera_offset.y*csy_sc; max_r=max(r1,r2,r3)
            if -max_r<csx<SW+max_r:
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx),int(csy)),r1)
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx+r1*.7),int(csy+5)),r2)
                pygame.draw.circle(screen,CLOUD_WHITE,(int(csx-r1*.6),int(csy+3)),r3)

        draw_pymunk_static_shapes(screen, space, camera_offset)
        draw_dynamic_shapes(screen, space, camera_offset, truck)
        truck.draw(screen, camera_offset)

        timer_str = f"Time: {level_time_taken:.2f}" if level_finished else f"Time: {(time.time()-level_start_time):.2f}"
        timer_surf = font.render(timer_str,True,WHITE); screen.blit(timer_surf,(10,10))
        level_surf = font.render(f"Level: {current_level}",True,WHITE); screen.blit(level_surf,(SW-level_surf.get_width()-10,10))

        if level_finished: # This implies level_complete_flag[0] is true
            if game_over and current_level == max_level: # Specifically for completing the last level
                msg_text = "All Levels Complete!"
                prompt_text = "R: Restart Level 1, ESC: Exit"
            else: # For completing any level that is not the last one
                msg_text = "Level Complete!"
                prompt_text = "N: Next Level, R: Retry This Level"

            msg_surf = font.render(msg_text,True,YELLOW); prompt_surf = small_font.render(prompt_text,True,WHITE)
            msg_rect = msg_surf.get_rect(center=(SW//2,SH//2-20)); prompt_rect = prompt_surf.get_rect(center=(SW//2,SH//2+20))
            screen.blit(msg_surf,msg_rect); screen.blit(prompt_surf,prompt_rect)

        pygame.display.flip()
    pygame.quit(); sys.exit()

if __name__ == "__main__":
    main()