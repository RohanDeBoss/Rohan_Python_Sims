import pygame
import pymunk
import math
import sys
import time
import random

# (Constants and Truck class remain the same as your previous version)
# --- Constants ---
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 600
FPS = 60

# Colors
WHITE = (255, 255, 255); BLACK = (0, 0, 0)
RED = (214, 73, 54); DARK_RED = (179, 58, 45)
BASE_SKY_BLUE = (135, 206, 235); PURPLE = (103, 58, 132)
YELLOW = (255, 193, 7); ORANGE = (255, 87, 34)
DARK_GREEN = (60, 130, 50); DEBUG_MAGENTA = (255, 0, 255)
GRAY = (128, 128, 128); WINDOW_BLUE = (100, 140, 200)
SKY_HORIZON_BLUE = (100, 180, 220); HILL_GREEN_FAR = (50, 120, 40)
HILL_GREEN_NEAR = (80, 160, 70); CLOUD_WHITE = (240, 240, 250)
PLATFORM_BLUE = (60, 100, 180)
ICE_BLUE = (173, 216, 230, 220)

# Physics Constants
GRAVITY = (0, -900)
CHASSIS_MASS = 32; WHEEL_MASS = 1.5
CHASSIS_FRICTION = 0.6; WHEEL_FRICTION = 1.5
GROUND_FRICTION = 0.95; CHASSIS_ELASTICITY = 0.25
WHEEL_ELASTICITY = 0.18; GROUND_ELASTICITY = 0.2
OBSTACLE_FRICTION = 0.5; OBSTACLE_ELASTICITY = 0.5
ICE_FRICTION = 0.1
ICE_ELASTICITY = 0.1

# Control Forces / Motor Parameters
MOTOR_TARGET_RATE_FORWARD = 26; MOTOR_TARGET_RATE_BACKWARD = -26
MOTOR_MAX_FORCE = 200000; BRAKE_MAX_FORCE = MOTOR_MAX_FORCE * 0.9
ROTATION_TORQUE = 1300000

MAX_ANGULAR_VELOCITY_CHASSIS = 10
ANGULAR_DRAG_COEFFICIENT = 5000
ANGULAR_DRAG_POWER = 1

# Collision Types
COLLISION_TYPE_CHASSIS = 1; COLLISION_TYPE_WHEEL = 2
COLLISION_TYPE_GROUND = 3; COLLISION_TYPE_OBSTACLE = 4
COLLISION_TYPE_FINISH = 5; COLLISION_TYPE_PLATFORM = 6
COLLISION_TYPE_ICE = 7

GROUND_BOTTOM_Y_PYMUNK = -1000
KILL_Y_COORDINATE = -300

BG_DISTANT_SCROLL_X = 0.05; BG_DISTANT_SCROLL_Y = 0.01
BG_MID_SCROLL_X = 0.2; BG_MID_SCROLL_Y = 0.03
BG_NEAR_SCROLL_X = 0.4; BG_NEAR_SCROLL_Y = 0.08
SH = SCREEN_HEIGHT; SW = SCREEN_WIDTH
kinematic_platforms_data = []


class Truck: # (Same as before)
    def __init__(self, space, pos):
        self.space = space
        self.chassis_dims = (71, 30)
        self.wheel_radius = 17
        moment = pymunk.moment_for_box(CHASSIS_MASS, self.chassis_dims)
        self.chassis_body = pymunk.Body(CHASSIS_MASS, moment)
        self.chassis_body.position = pos
        self.chassis_shape = pymunk.Poly.create_box(self.chassis_body, self.chassis_dims, radius=4.0)
        self.chassis_shape.friction = CHASSIS_FRICTION; self.chassis_shape.elasticity = CHASSIS_ELASTICITY
        self.chassis_shape.collision_type = COLLISION_TYPE_CHASSIS
        self.space.add(self.chassis_body, self.chassis_shape)
        wheel_offset_x = self.chassis_dims[0] * 0.38; chassis_half_height = self.chassis_dims[1] / 2
        wheel_axle_y_offset_from_center = -chassis_half_height - 1
        self.wheel_positions_local = [pymunk.Vec2d(-wheel_offset_x, wheel_axle_y_offset_from_center), pymunk.Vec2d(wheel_offset_x, wheel_axle_y_offset_from_center)]
        self.wheel_bodies = []; self.wheel_shapes = []; self.wheel_motors = []; self.wheel_joints = []
        for i, local_axle_pos in enumerate(self.wheel_positions_local):
            wheel_moment = pymunk.moment_for_circle(WHEEL_MASS,0,self.wheel_radius); wheel_body=pymunk.Body(WHEEL_MASS,wheel_moment)
            wheel_body.position = self.chassis_body.local_to_world(local_axle_pos)
            wheel_shape = pymunk.Circle(wheel_body,self.wheel_radius); wheel_shape.friction=WHEEL_FRICTION; wheel_shape.elasticity=WHEEL_ELASTICITY
            wheel_shape.collision_type=COLLISION_TYPE_WHEEL; wheel_shape.color=GRAY
            self.space.add(wheel_body,wheel_shape); self.wheel_bodies.append(wheel_body); self.wheel_shapes.append(wheel_shape)
            pivot=pymunk.PivotJoint(self.chassis_body,wheel_body,local_axle_pos,(0,0)); pivot.collide_bodies=False; self.space.add(pivot); self.wheel_joints.append(pivot)
            motor=pymunk.SimpleMotor(self.chassis_body,wheel_body,0); motor.max_force=0; self.space.add(motor); self.wheel_motors.append(motor)
        self.initial_pos=pymunk.Vec2d(pos.x,pos.y); self.initial_angle=0.0; self.reset()
    def reset(self):
        self.chassis_body.position=self.initial_pos; self.chassis_body.angle=self.initial_angle; self.chassis_body.velocity=(0,0); self.chassis_body.angular_velocity=0
        for i,wb in enumerate(self.wheel_bodies):
            wb.position=self.chassis_body.local_to_world(self.wheel_positions_local[i]); wb.angle=self.chassis_body.angle; wb.velocity=(0,0); wb.angular_velocity=0
            self.wheel_motors[i].rate=0; self.wheel_motors[i].max_force=0
    def drive_forward(self): [(setattr(m,'rate',MOTOR_TARGET_RATE_FORWARD),setattr(m,'max_force',MOTOR_MAX_FORCE)) for m in self.wheel_motors]
    def drive_backward(self): [(setattr(m,'rate',MOTOR_TARGET_RATE_BACKWARD),setattr(m,'max_force',MOTOR_MAX_FORCE)) for m in self.wheel_motors]
    def apply_torque_nose_down(self): self.chassis_body.torque-=ROTATION_TORQUE
    def apply_torque_nose_up(self): self.chassis_body.torque+=ROTATION_TORQUE
    def release_throttle_and_brake(self): [(setattr(m,'rate',0),setattr(m,'max_force',BRAKE_MAX_FORCE/5)) for m in self.wheel_motors]
    def draw(self,screen,camera_offset):
        co=camera_offset; ch_verts_w=[self.chassis_body.local_to_world(v) for v in self.chassis_shape.get_vertices()]
        scr_pts_ch=[(int(p.x-co.x),int(SH-(p.y-co.y))) for p in ch_verts_w]
        if len(scr_pts_ch)>=3:pygame.draw.polygon(screen,RED,scr_pts_ch)
        ch_hw,ch_hh=self.chassis_dims[0]/2.0,self.chassis_dims[1]/2.0
        cab_width=(ch_hw-5)-2; cab_front_local_x=ch_hw-0; cab_back_local_x=cab_front_local_x-cab_width
        cab_bottom_y=ch_hh; cab_top_front_y=cab_bottom_y+18; cab_top_back_y=cab_bottom_y+25
        cab_local_pts=[pymunk.Vec2d(cab_back_local_x,cab_bottom_y),pymunk.Vec2d(cab_front_local_x,cab_bottom_y),pymunk.Vec2d(cab_front_local_x,cab_top_front_y),pymunk.Vec2d(cab_back_local_x,cab_top_back_y)]
        win_ih,win_iv=5,5; win_back_x=cab_back_local_x+win_ih; win_front_x=cab_front_local_x-win_ih
        win_bottom_y=cab_bottom_y+win_iv; win_top_y_on_front=cab_top_front_y-win_iv; win_top_y_on_back=cab_top_back_y-win_iv
        window_local_pts=[pymunk.Vec2d(win_back_x,win_bottom_y),pymunk.Vec2d(win_front_x,win_bottom_y),pymunk.Vec2d(win_front_x,win_top_y_on_front),pymunk.Vec2d(win_back_x,win_top_y_on_back)]
        cab_verts_w=[self.chassis_body.local_to_world(p) for p in cab_local_pts]; scr_pts_cab=[(int(p.x-co.x),int(SH-(p.y-co.y))) for p in cab_verts_w]
        if len(scr_pts_cab)>=3:pygame.draw.polygon(screen,DARK_RED,scr_pts_cab)
        win_verts_w=[self.chassis_body.local_to_world(p) for p in window_local_pts]; scr_pts_win=[(int(p.x-co.x),int(SH-(p.y-co.y))) for p in win_verts_w]
        if len(scr_pts_win)>=3:pygame.draw.polygon(screen,WINDOW_BLUE,scr_pts_win)
        if len(scr_pts_ch)>=3:pygame.draw.polygon(screen,BLACK,scr_pts_ch,1)
        if len(scr_pts_cab)>=3:pygame.draw.polygon(screen,BLACK,scr_pts_cab,1)
        for i,wb in enumerate(self.wheel_bodies):
            wp_w=wb.position;sx,sy=int(wp_w.x-co.x),int(SH-(wp_w.y-co.y));wc=getattr(self.wheel_shapes[i],'color',GRAY)
            pygame.draw.circle(screen,wc,(sx,sy),self.wheel_radius);pygame.draw.circle(screen,BLACK,(sx,sy),self.wheel_radius,1)
            sp_end_w=wb.local_to_world((self.wheel_radius*0.9,0));sp_esx,sp_esy=int(sp_end_w.x-co.x),int(SH-(sp_end_w.y-co.y))
            pygame.draw.line(screen,WHITE,(sx,sy),(sp_esx,sp_esy),2)

# (Helper functions create_static_segment, create_box, create_dynamic_circle, create_finish_line, create_kinematic_platform are same)
def create_static_segment(space,p1,p2,thickness=6,friction=GROUND_FRICTION,elasticity=GROUND_ELASTICITY,collision_type=COLLISION_TYPE_GROUND,color=DARK_GREEN):
    shape=pymunk.Segment(space.static_body,p1,p2,thickness/2.0);shape.friction=friction;shape.elasticity=elasticity;shape.collision_type=collision_type;shape.color=color;space.add(shape);return shape
def create_box(space,pos,size,mass=None,angle=0,friction=OBSTACLE_FRICTION,elasticity=OBSTACLE_ELASTICITY,collision_type=COLLISION_TYPE_OBSTACLE,color=ORANGE,radius=0.5):
    if mass is None:body=space.static_body;hw,hh=size[0]/2,size[1]/2;lpts=[(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)];xfm=pymunk.Transform.identity().translated(pos.x,pos.y).rotated(math.radians(angle));shape=pymunk.Poly(body,lpts,transform=xfm,radius=radius)
    else:body=pymunk.Body(mass,pymunk.moment_for_box(mass,size));body.position=pos;body.angle=math.radians(angle);shape=pymunk.Poly.create_box(body,size,radius=radius);space.add(body)
    shape.friction=friction;shape.elasticity=elasticity;shape.collision_type=collision_type;shape.color=color;space.add(shape);return (body,shape) if mass is not None else (None,shape)
def create_dynamic_circle(space,pos,radius,mass,**kwargs):
    defaults={'friction':OBSTACLE_FRICTION,'elasticity':OBSTACLE_ELASTICITY,'collision_type':COLLISION_TYPE_OBSTACLE,'color':ORANGE};props={**defaults,**kwargs};body=pymunk.Body(mass,pymunk.moment_for_circle(mass,0,radius));body.position=pos;shp=pymunk.Circle(body,radius)
    shp.friction=props['friction'];shp.elasticity=props['elasticity'];shp.collision_type=props['collision_type'];shp.color=props['color'];space.add(body,shp);return body,shp
def create_finish_line(space,pos,width=20,height=80):
    hw,hh=width/2,height/2;pts=[(-hw,-hh),(-hw,hh),(hw,hh),(hw,-hh)];xfm=pymunk.Transform(tx=pos.x,ty=pos.y);shp=pymunk.Poly(space.static_body,pts,transform=xfm);shp.sensor=True;shp.collision_type=COLLISION_TYPE_FINISH;shp.color=YELLOW;space.add(shp);return shp
def create_stairs(space,start_bottom_left_pos,step_width,step_height,num_steps,direction=1,**kwargs):
    stair_color=kwargs.pop('color',ORANGE);stair_friction=kwargs.pop('friction',OBSTACLE_FRICTION);stair_elasticity=kwargs.pop('elasticity',OBSTACLE_ELASTICITY);current_x=start_bottom_left_pos.x;current_y=start_bottom_left_pos.y
    for i in range(num_steps):
        scx=current_x+(step_width/2*direction);scy=current_y+(step_height/2)
        create_box(space,pymunk.Vec2d(scx,scy),(step_width,step_height),mass=None,color=stair_color,friction=stair_friction,elasticity=stair_elasticity,**kwargs)
        current_x+=step_width*direction;current_y+=step_height
def create_kinematic_platform(space,initial_pos,size,vertical_travel,speed,**kwargs):
    body=pymunk.Body(body_type=pymunk.Body.KINEMATIC);body.position=initial_pos;shape=pymunk.Poly.create_box(body,size,radius=0.5)
    shape.friction=kwargs.get('friction',0.8);shape.elasticity=kwargs.get('elasticity',0.1);shape.collision_type=COLLISION_TYPE_PLATFORM;shape.color=kwargs.get('color',PLATFORM_BLUE);space.add(body,shape)
    platform_data={'body':body,'min_y':initial_pos.y,'max_y':initial_pos.y+vertical_travel,'current_speed':speed,'initial_speed':speed};kinematic_platforms_data.append(platform_data);return body,shape

# (Drawing functions and clear_level are same)
def draw_pymunk_static_shapes(screen,space,camera_offset):
    co=camera_offset
    for shape in space.static_body.shapes:
        color=getattr(shape,'color',DEBUG_MAGENTA);draw_color=color[:3] if len(color)==4 else color
        if isinstance(shape,pymunk.Segment):
            p1w,p2w=shape.a,shape.b
            if hasattr(shape,'collision_type') and shape.collision_type==COLLISION_TYPE_GROUND:
                seg_c=draw_color;pb1=pymunk.Vec2d(p1w.x,GROUND_BOTTOM_Y_PYMUNK);pb2=pymunk.Vec2d(p2w.x,GROUND_BOTTOM_Y_PYMUNK)
                poly_pts=[(int(p1w.x-co.x),int(SH-(p1w.y-co.y))),(int(p2w.x-co.x),int(SH-(p2w.y-co.y))),(int(pb2.x-co.x),int(SH-(pb2.y-co.y))),(int(pb1.x-co.x),int(SH-(pb1.y-co.y)))]
                if len(set(poly_pts))>=3:
                    try:pygame.draw.polygon(screen,seg_c,poly_pts)
                    except ValueError:print(f"Warn:CollGroundPoly:{poly_pts}")
                pygame.draw.line(screen,BLACK,poly_pts[0],poly_pts[1],1)
            else:
                rad=max(1,int(shape.radius));sp1=(int(p1w.x-co.x),int(SH-(p1w.y-co.y)));sp2=(int(p2w.x-co.x),int(SH-(p2w.y-co.y)))
                if sp1!=sp2:
                    try:pygame.draw.line(screen,draw_color,sp1,sp2,int(rad*2))
                    except ValueError:print(f"Warn:InvLine:{sp1},{sp2},{int(rad*2)}")
        elif isinstance(shape,pymunk.Poly):
            poly_c,poly_w=draw_color,0
            if shape.sensor and shape.collision_type==COLLISION_TYPE_FINISH:poly_c=YELLOW
            verts_w=[shape.body.local_to_world(v) for v in shape.get_vertices()]
            scr_pts=[(int(p.x-co.x),int(SH-(p.y-co.y))) for p in verts_w]
            if len(scr_pts)>=3:
                try:pygame.draw.polygon(screen,poly_c,scr_pts,poly_w)
                except ValueError:print(f"Warn:CollStaticPoly:{scr_pts}")
                if poly_w==0 and not (shape.sensor and shape.collision_type==COLLISION_TYPE_FINISH):
                    try:pygame.draw.polygon(screen,BLACK,scr_pts,1)
                    except ValueError:pass
def draw_dynamic_shapes(screen,space,camera_offset,truck):
    co=camera_offset;ignore_b=[truck.chassis_body]+truck.wheel_bodies
    for body in space.bodies:
        if body.body_type in (pymunk.Body.DYNAMIC,pymunk.Body.KINEMATIC) and body not in ignore_b:
            for shp in body.shapes:
                color=getattr(shp,'color',ORANGE);draw_color=color[:3] if len(color)==4 else color
                if isinstance(shp,pymunk.Circle):
                    pos_w=body.local_to_world(shp.offset);sx,sy=int(pos_w.x-co.x),int(SH-(pos_w.y-co.y));rad=int(shp.radius)
                    if rad>0:pygame.draw.circle(screen,draw_color,(sx,sy),rad);pygame.draw.circle(screen,BLACK,(sx,sy),rad,1)
                elif isinstance(shp,pymunk.Poly):
                    verts_w=[body.local_to_world(v) for v in shp.get_vertices()];scr_pts=[(int(p.x-co.x),int(SH-(p.y-co.y))) for p in verts_w]
                    if len(scr_pts)>=3:
                        try:pygame.draw.polygon(screen,draw_color,scr_pts,0);pygame.draw.polygon(screen,BLACK,scr_pts,1)
                        except ValueError:print(f"Warn:CollDynPoly:{scr_pts}")
def clear_level(space,truck):
    global kinematic_platforms_data;kinematic_platforms_data.clear()
    bodies_to_remove=[b for b in space.bodies if b.body_type!=pymunk.Body.STATIC and b!=truck.chassis_body and b not in truck.wheel_bodies]
    for body in bodies_to_remove:
        for c in list(body.constraints):
             if c in space.constraints:space.remove(c)
        for s in list(body.shapes):
            if s in space.shapes:space.remove(s)
        if body in space.bodies:space.remove(body)
    for s in list(space.static_body.shapes):
        if s in space.shapes:space.remove(s)

# --- LEVEL DESIGNS ---
# (Constants and Truck class, other helper functions, main loop remain the same as your previous version)

# --- LEVEL DESIGNS ---
def load_level(space, level_index):
    print(f"Loading Level {level_index}...")
    base_y = 150
    spawn_height_offset = 60
    start_pos = pymunk.Vec2d(150, base_y + spawn_height_offset)
    ground_at_finish_offset = -45
    normal_stair_width = 55
    icy_stair_width = 65

    def cgp(points, **kwargs):
        for i in range(len(points) - 1): create_static_segment(space, points[i], points[i+1], **kwargs)

    if level_index == 1: # Standard intro
        theme_color=ORANGE;finish_pos=pymunk.Vec2d(2800,base_y+60);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        gp=[(-200,base_y),(600,base_y),(800,base_y+20),(1100,base_y+30),(1400,base_y),(1700,base_y-10),(2000,base_y-10),(2300,base_y+40),(finish_pos.x,gnd_y_fin),(finish_pos.x+200,gnd_y_fin)]
        cgp(gp); create_box(space,pymunk.Vec2d(950,base_y+55),(50,50),mass=5,color=theme_color); create_box(space,pymunk.Vec2d(1800,base_y+15),(100,30),mass=8,color=theme_color); create_box(space,pymunk.Vec2d(2450,base_y+65),(60,40),mass=6,color=theme_color)
    elif level_index == 2: # Jumps
        theme_color=YELLOW;start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset);finish_pos=pymunk.Vec2d(3000,base_y+120);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        gp_section1 = [(-100,base_y),(300,base_y),(600,base_y+80),(900,base_y+80)]; cgp(gp_section1)
        gp_section2 = [(1050,base_y+80),(1100,base_y+20),(1400,base_y+30),(1600,base_y+100)]; cgp(gp_section2)
        gp_section3 = [(1750,base_y+100),(2100,base_y+40),(2400,base_y+40),(2700,base_y+120),(finish_pos.x,gnd_y_fin),(finish_pos.x+100,gnd_y_fin)]; cgp(gp_section3)
        create_box(space,pymunk.Vec2d(750,base_y+105),(40,40),mass=4,color=theme_color); create_dynamic_circle(space,pymunk.Vec2d(1250,base_y+80),25,mass=5,color=theme_color); create_box(space,pymunk.Vec2d(1950,base_y+65),(70,30),mass=7,color=theme_color); create_box(space,pymunk.Vec2d(2500,base_y+150),(50,50),mass=6,color=theme_color); create_dynamic_circle(space,pymunk.Vec2d(2200,base_y+80),20,mass=4,color=RED)
    elif level_index == 3: # Tricky
        theme_color=PURPLE
        start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset-20)
        finish_pos=pymunk.Vec2d(3400,base_y+20)
        gnd_y_fin=finish_pos.y+ground_at_finish_offset
        gp=[(-100,base_y),(200,base_y),(350,base_y-40),(500,base_y-40),(650,base_y+60),(800,base_y+60),(900,base_y+10),(1200,base_y+10),(1400,base_y-30),(1700,base_y-30),(1900,base_y+70),(2100,base_y+80),(2200,base_y),(2500,base_y),(2700,base_y-50),(3100,base_y-50),(finish_pos.x,gnd_y_fin),(finish_pos.x+100,gnd_y_fin)]
        cgp(gp)
        create_box(space,pymunk.Vec2d(425,base_y-30),(100,20),mass=None,angle=-20,color=theme_color)
        create_box(space, pymunk.Vec2d(1050, base_y + 40), (20, 73), mass=None, color=theme_color)
        create_box(space,pymunk.Vec2d(1800,base_y+5),(80,80),mass=15,color=theme_color)
        create_box(space,pymunk.Vec2d(2350,base_y+50),(100,100),mass=20,color=theme_color)
        create_dynamic_circle(space,pymunk.Vec2d(2850,base_y+30),30,mass=8,color=theme_color)

    elif level_index == 4: # Bridge/Jump Focus - BRIDGE SUPPORTS ADDED
        theme_color=RED;start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset+30);finish_pos=pymunk.Vec2d(3600,base_y+70);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        gap_width = 200
        bridge_plank_width = 190
        plank_height = 15
        support_ledge_width = 28 # How wide the static supports are

        # Y-levels for the top surface of the static platforms forming the gap edges
        platform1_top_y = base_y + 100
        platform2_top_y = base_y + 120

        # Ground Section 1: Before first gap
        gp1 = [(-100, base_y + 40), (600 - support_ledge_width/2, platform1_top_y)] # End just before support
        cgp(gp1)
        create_box(space, pymunk.Vec2d(600, platform1_top_y - plank_height/2 -1 ), (support_ledge_width, plank_height+2), mass=None, color=DARK_GREEN) # Support 1A

        # Ground Section 2: Between gaps
        gp2_start_x = 600 + gap_width
        gp2 = [(gp2_start_x + support_ledge_width/2, platform1_top_y), # Start just after support
               (1200, base_y + 60), (1500, base_y + 60),
               (1700 - support_ledge_width/2, platform2_top_y)] # End just before support
        cgp(gp2)
        create_box(space, pymunk.Vec2d(gp2_start_x, platform1_top_y - plank_height/2 -1), (support_ledge_width, plank_height+2), mass=None, color=DARK_GREEN) # Support 1B
        create_box(space, pymunk.Vec2d(1700, platform2_top_y - plank_height/2 -1), (support_ledge_width, plank_height+2), mass=None, color=DARK_GREEN) # Support 2A


        # Ground Section 3: After second gap to finish
        gp3_start_x = 1700 + gap_width
        gp3 = [(gp3_start_x + support_ledge_width/2, platform2_top_y), # Start just after support
               (2400, base_y + 80), (2700, base_y + 80), (2900, base_y + 100),
               (3200, base_y + 100), (finish_pos.x, gnd_y_fin), (finish_pos.x + 100, gnd_y_fin)]
        cgp(gp3)
        create_box(space, pymunk.Vec2d(gp3_start_x, platform2_top_y - plank_height/2 -1), (support_ledge_width, plank_height+2), mass=None, color=DARK_GREEN) # Support 2B


        # Bridge Planks - Y position needs to be ON TOP of where supports end.
        # Supports are at platform_top_y. Planks are plank_height tall.
        # So plank center Y should be platform_top_y + plank_height/2
        plank_center_y1 = platform1_top_y + plank_height / 2
        plank_center_y2 = platform2_top_y + plank_height / 2

        create_box(space, pymunk.Vec2d(600 + gap_width / 2, plank_center_y1), (bridge_plank_width, plank_height), mass=10, color=theme_color)
        create_box(space, pymunk.Vec2d(600 + gap_width / 2, plank_center_y1 + plank_height / 2 + 10 / 2), (20, 10), mass=1, color=ORANGE)

        create_box(space, pymunk.Vec2d(1700 + gap_width / 2, plank_center_y2), (bridge_plank_width, plank_height), mass=10, color=theme_color)

        create_dynamic_circle(space, pymunk.Vec2d(1600, base_y + 200), 30, mass=8, color=PURPLE)
        create_box(space, pymunk.Vec2d(1900 + 50, platform2_top_y + 15), (30, 30), mass=3, color=ORANGE)
        create_dynamic_circle(space, pymunk.Vec2d(2850, base_y + 150), 40, mass=12, color=ORANGE)
        create_box(space, pymunk.Vec2d(3100, base_y + 125), (60, 60), mass=12, color=theme_color)


    elif level_index == 5: # Varied - RED BOX HEIGHT REDUCED
        theme_color=DARK_GREEN;lvl5_base_y=200;start_pos=pymunk.Vec2d(100,lvl5_base_y+spawn_height_offset);finish_pos=pymunk.Vec2d(4200,lvl5_base_y+100);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        path=[(-100,lvl5_base_y)];cx,cy=-100,lvl5_base_y;min_y_allowed=lvl5_base_y-50
        sections_data=[(random.randint(550,700),random.randint(15,30)), (random.randint(450,600),random.randint(-30,-15)),
                       (random.randint(650,800),random.randint(40,60)), (random.randint(550,700),random.randint(-25,-10)),
                       (random.randint(750,900),random.randint(50,70)), (random.randint(650,800),random.randint(10,25))]
        for i, (dx,dy_change) in enumerate(sections_data): cx+=dx;target_y=cy+dy_change;cy=max(min_y_allowed,target_y);path.append((cx,cy))
        last_generated_x, last_generated_y = path[-1]; path.append((last_generated_x + 300, last_generated_y))
        path.append((finish_pos.x - 200, gnd_y_fin)); path.append((finish_pos.x, gnd_y_fin)); path.append((finish_pos.x + 200, gnd_y_fin))
        cgp(path,color=DARK_GREEN)
        create_box(space,pymunk.Vec2d(path[1][0]+200,path[1][1]+25),(100,40),mass=12,color=PURPLE)
        create_dynamic_circle(space,pymunk.Vec2d(path[2][0]+300,path[2][1]+34),34,mass=10,color=ORANGE)
        create_box(space,pymunk.Vec2d(path[4][0]+100,path[4][1]+15),(120,20),mass=7,angle=-5,color=PURPLE)
        static_box_segment_start_x=path[5][0];static_box_segment_start_y=path[5][1]
        create_box(space,pymunk.Vec2d(static_box_segment_start_x+150,static_box_segment_start_y+33),(70,80),mass=None,color=DARK_RED) # Reduced height to 75
        create_dynamic_circle(space,pymunk.Vec2d(path[6][0]+200,path[6][1]+30),30,mass=7,color=ORANGE)


    elif level_index == 6: # Stairs
        start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset-20);platform1_y=base_y+160;platform2_y=platform1_y+150
        finish_pos=pymunk.Vec2d(1800,platform2_y+30);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        cgp([(-100,base_y),(300,base_y)]);create_stairs(space,pymunk.Vec2d(300,base_y),normal_stair_width,20,8,1,color=GRAY)
        cgp([(300+normal_stair_width*8,platform1_y),(1000,platform1_y)]);create_stairs(space,pymunk.Vec2d(1000,platform1_y),normal_stair_width-5,25,6,1,color=GRAY)
        cgp([(1000+(normal_stair_width-5)*6,platform2_y),(finish_pos.x,gnd_y_fin),(finish_pos.x+200,gnd_y_fin)])
        create_box(space,pymunk.Vec2d(850,platform1_y+30),(40,40),mass=5,color=ORANGE)

    elif level_index == 7: # Elevators - higher travel & faster
        start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset-20);finish_pos=pymunk.Vec2d(2500,base_y+150);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        cgp([(-100,base_y),(400,base_y)])
        create_kinematic_platform(space,pymunk.Vec2d(500,base_y),(150,20),170,60)
        elevated_y1=base_y+170;cgp([(700,elevated_y1),(1000,elevated_y1)]);cgp([(1150,elevated_y1-40),(1500,elevated_y1-40)])
        second_elevator_start_y=elevated_y1-40;second_elevator_travel=gnd_y_fin-second_elevator_start_y+60
        create_kinematic_platform(space,pymunk.Vec2d(1600,second_elevator_start_y),(120,20),second_elevator_travel,55)
        cgp([(1800,gnd_y_fin),(finish_pos.x,gnd_y_fin),(finish_pos.x+200,gnd_y_fin)])
        create_dynamic_circle(space,pymunk.Vec2d(850,elevated_y1+30),20,mass=3,color=RED)

    elif level_index == 8: # Ice and Jumps - REVISED ENDING
        start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset)
        finish_y_climb = base_y + 180 # Make finish higher for a climb
        finish_pos=pymunk.Vec2d(3200, finish_y_climb)
        gnd_y_fin=finish_pos.y+ground_at_finish_offset

        cgp([(-100,base_y),(300,base_y)]);cgp([(300,base_y),(600,base_y+80)],friction=ICE_FRICTION,color=ICE_BLUE);cgp([(600,base_y+80),(900,base_y+80)],friction=ICE_FRICTION,color=ICE_BLUE)
        create_box(space,pymunk.Vec2d(750,base_y+80+25),(40,40),mass=5,color=ORANGE,friction=0.3);cgp([(900,base_y+80),(1000,base_y+100)],friction=ICE_FRICTION,color=ICE_BLUE)
        cgp([(1200,base_y+70),(1400,base_y+60)],friction=ICE_FRICTION,color=ICE_BLUE);cgp([(1400,base_y+60),(1800,base_y+60)],friction=ICE_FRICTION,color=ICE_BLUE)
        cgp([(1800,base_y+60),(1900,base_y+40)]);cgp([(2050,base_y+40),(2300,base_y+40)])
        # Start the climb
        climb_start_x = 2300; climb_start_y = base_y + 40
        climb_mid_x = climb_start_x + 400; climb_mid_y = climb_start_y + 80 # Mid-point of climb
        finish_approach_x = finish_pos.x - 200 # Start of flat section before finish
        cgp([(climb_start_x, climb_start_y), (climb_mid_x, climb_mid_y)]) # First part of climb
        cgp([(climb_mid_x, climb_mid_y), (finish_approach_x, gnd_y_fin)]) # Second part of climb to finish height
        cgp([(finish_approach_x, gnd_y_fin), (finish_pos.x, gnd_y_fin), (finish_pos.x + 200, gnd_y_fin)]) # Flat to finish

    elif level_index == 9: # Arctic Expedition
        start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset);finish_pos=pymunk.Vec2d(3800,base_y+180);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        cgp([(-100,base_y),(400,base_y)])
        create_stairs(space,pymunk.Vec2d(400,base_y),icy_stair_width,15,5,1,color=ICE_BLUE,friction=ICE_FRICTION,elasticity=ICE_ELASTICITY)
        plat1_y=base_y+15*5; cgp([(400+icy_stair_width*5,plat1_y),(900,plat1_y)])
        cgp([(900,plat1_y),(1100,plat1_y-20)],friction=ICE_FRICTION,color=ICE_BLUE)
        cgp([(1250,plat1_y-40),(1500,plat1_y-40)],friction=ICE_FRICTION,color=ICE_BLUE)
        chasm_floor_y=base_y-100
        create_static_segment(space,(1500,plat1_y-40),(1500,chasm_floor_y),color=ICE_BLUE[:3]);create_static_segment(space,(1800,plat1_y-40),(1800,chasm_floor_y),color=ICE_BLUE[:3]);create_static_segment(space,(1500,chasm_floor_y),(1800,chasm_floor_y),friction=ICE_FRICTION,color=ICE_BLUE)
        create_kinematic_platform(space,pymunk.Vec2d(1600,plat1_y-40+10),(150,20),110,60)
        cgp([(1800,plat1_y-40),(2100,plat1_y-40)])
        current_x,current_y=2100,plat1_y-40
        for i in range(3):
            cgp([(current_x,current_y),(current_x+150,current_y+10)],friction=ICE_FRICTION,color=ICE_BLUE);current_x+=150+100;current_y+=10-5
            cgp([(current_x,current_y),(current_x+50,current_y)],friction=ICE_FRICTION,color=ICE_BLUE);current_x+=50
        cgp([(current_x,current_y),(current_x+300,gnd_y_fin-20)]);cgp([(current_x+300,gnd_y_fin-20),(finish_pos.x,gnd_y_fin),(finish_pos.x+200,gnd_y_fin)])

    else: # Placeholder
        ln=level_index-9;start_pos=pymunk.Vec2d(100,base_y+spawn_height_offset);finish_y_val=base_y+(ln%3)*40+40;finish_pos=pymunk.Vec2d(1500+ln*200,finish_y_val);gnd_y_fin=finish_pos.y+ground_at_finish_offset
        path=[(-100,base_y)];cx=-100
        for _ in range(3+ln%2):cx+=random.randint(400,600);path.append((cx,base_y+random.randint(-20,30)))
        path.append((finish_pos.x,gnd_y_fin));path.append((finish_pos.x+100,gnd_y_fin));cgp(path)
        for i in range(1+ln%3):ox=400+i*(400+ln*30);oy=base_y+25+random.randint(0,40);create_box(space,pymunk.Vec2d(ox,oy),(40,50),mass=5+i*2,color=ORANGE)

    finish_shape=create_finish_line(space,finish_pos)
    print(f"Level {level_index} loaded. Start: {start_pos}, Finish: {(finish_pos.x,finish_pos.y)}, Ground @ Finish X: {gnd_y_fin if 'gnd_y_fin' in locals() else 'N/A'}")
    return start_pos,finish_shape


def main():
    pygame.init(); screen=pygame.display.set_mode((SW,SH)); pygame.display.set_caption("Jelly Truck Adventures - Level Up!"); clock=pygame.time.Clock()
    font=pygame.font.Font(None,36);small_font=pygame.font.Font(None,24); space=pymunk.Space();space.gravity=GRAVITY
    current_level=1; max_level=9; truck=Truck(space,pymunk.Vec2d(150,250))
    level_finished=False;game_over=False;level_start_time=time.time();level_time_taken=0;level_complete_flag=[False]
    bg_elements=[{'y_on_screen_top':0,'height_on_screen':int(SH*.6),'color':SKY_HORIZON_BLUE,'scroll_x':BG_DISTANT_SCROLL_X,'scroll_y':.01}, {'y_on_screen_top':int(SH*.5),'height_on_screen':int(SH*.3),'color':HILL_GREEN_FAR,'scroll_x':BG_MID_SCROLL_X,'scroll_y':BG_MID_SCROLL_Y}, {'y_on_screen_top':int(SH*.7),'height_on_screen':int(SH*.3),'color':HILL_GREEN_NEAR,'scroll_x':BG_NEAR_SCROLL_X,'scroll_y':BG_NEAR_SCROLL_Y}]
    clouds=[];[clouds.append([random.randint(-SW,SW*2),random.randint(int(SH*.1),int(SH*.4)),random.randint(20,40),random.randint(15,35),random.randint(10,30)]) for _ in range(10)]
    def finish_collision_handler(arbiter,space,data):
        is_chassis=any(hasattr(s,'collision_type') and s.collision_type==COLLISION_TYPE_CHASSIS for s in arbiter.shapes)
        if is_chassis and not level_complete_flag[0]:print("Level Complete!");level_complete_flag[0]=True
        return True
    h_chassis=space.add_collision_handler(COLLISION_TYPE_CHASSIS,COLLISION_TYPE_FINISH);h_chassis.begin=finish_collision_handler
    camera_offset=pymunk.Vec2d(0,0);camera_smoothing=0.08
    def setup_new_level(level_idx):
        nonlocal level_start_time,level_finished,game_over,level_complete_flag,level_time_taken,truck,camera_offset
        clear_level(space,truck); start_pos_pymunk,_=load_level(space,level_idx); truck.initial_pos=start_pos_pymunk;truck.reset()
        camera_offset=pymunk.Vec2d(truck.chassis_body.position.x-SW/2,truck.chassis_body.position.y-SH/2)
        level_start_time=time.time();level_complete_flag[0]=False;level_finished=False;level_time_taken=0;game_over=False
    setup_new_level(current_level)
    running=True
    while running:
        dt=min(clock.tick(FPS)/1000.0,0.033)
        for event in pygame.event.get():
            if event.type==pygame.QUIT:running=False
            if event.type==pygame.KEYDOWN:
                if event.key==pygame.K_ESCAPE:running=False
                if event.key==pygame.K_r:
                    if game_over and not (level_finished and current_level==max_level):current_level=1;game_over=False
                    setup_new_level(current_level)
                if event.key==pygame.K_n and level_finished and not (current_level==max_level and game_over):
                    current_level+=1
                    if current_level>max_level:game_over=True
                    else:setup_new_level(current_level)
            if event.type==pygame.KEYUP:
                if not level_finished and not game_over:
                    if event.key in (pygame.K_UP,pygame.K_w,pygame.K_DOWN,pygame.K_s):truck.release_throttle_and_brake()
        keys=pygame.key.get_pressed()
        if not level_finished and not game_over:
            drive_action=False
            if keys[pygame.K_UP]or keys[pygame.K_w]:truck.drive_forward();drive_action=True
            if keys[pygame.K_DOWN]or keys[pygame.K_s]:truck.drive_backward();drive_action=True
            if not drive_action:truck.release_throttle_and_brake()
            if keys[pygame.K_LEFT]or keys[pygame.K_a]:truck.apply_torque_nose_up()
            if keys[pygame.K_RIGHT]or keys[pygame.K_d]:truck.apply_torque_nose_down()
            av=truck.chassis_body.angular_velocity
            if abs(av)>0.01:drag_torque=-ANGULAR_DRAG_COEFFICIENT*(abs(av)**ANGULAR_DRAG_POWER)*math.copysign(1,av);truck.chassis_body.torque+=drag_torque
            max_spin=MAX_ANGULAR_VELOCITY_CHASSIS
            if av>max_spin:truck.chassis_body.angular_velocity=max_spin
            elif av<-max_spin:truck.chassis_body.angular_velocity=-max_spin
        for plat_data in kinematic_platforms_data:
            body=plat_data['body'];current_y=body.position.y;current_speed=plat_data['current_speed']
            if current_y>=plat_data['max_y'] and current_speed>0:plat_data['current_speed']*=-1
            elif current_y<=plat_data['min_y'] and current_speed<0:plat_data['current_speed']*=-1
            body.velocity=(0,plat_data['current_speed'])
        space.step(dt)
        if truck.chassis_body.position.y<KILL_Y_COORDINATE and not level_finished and not game_over:
            print("Fell off map! Resetting level.");setup_new_level(current_level);continue
        if level_complete_flag[0] and not level_finished:
            level_finished=True;level_time_taken=time.time()-level_start_time;print(f"Internal:Level {current_level} Finished.Time:{level_time_taken:.2f}")
            if current_level==max_level:game_over=True
        target_cam_x=truck.chassis_body.position.x-SW/2;target_cam_y=truck.chassis_body.position.y-SH/2
        new_cam_x=camera_offset.x+(target_cam_x-camera_offset.x)*camera_smoothing;new_cam_y=camera_offset.y+(target_cam_y-camera_offset.y)*camera_smoothing
        camera_offset=pymunk.Vec2d(new_cam_x,new_cam_y)
        screen.fill(BASE_SKY_BLUE)
        for layer in bg_elements:
            lx=-(camera_offset.x*layer['scroll_x'])%SW;ay=layer['y_on_screen_top']-(camera_offset.y*layer['scroll_y'])
            pygame.draw.rect(screen,layer['color'],(lx,ay,SW,layer['height_on_screen']));pygame.draw.rect(screen,layer['color'],(lx+SW,ay,SW,layer['height_on_screen']));pygame.draw.rect(screen,layer['color'],(lx-SW,ay,SW,layer['height_on_screen']))
        csx_sc,csy_sc=BG_MID_SCROLL_X,BG_MID_SCROLL_Y
        for c_dat in clouds:
            bx,by,r1,r2,r3=c_dat;cwx=bx-camera_offset.x*csx_sc;sww=SW*1.5;csx=(cwx+SW/2)%sww-SW*.25;csy=by-camera_offset.y*csy_sc;max_r=max(r1,r2,r3)
            if -max_r<csx<SW+max_r:pygame.draw.circle(screen,CLOUD_WHITE,(int(csx),int(csy)),r1);pygame.draw.circle(screen,CLOUD_WHITE,(int(csx+r1*.7),int(csy+5)),r2);pygame.draw.circle(screen,CLOUD_WHITE,(int(csx-r1*.6),int(csy+3)),r3)
        draw_pymunk_static_shapes(screen,space,camera_offset);draw_dynamic_shapes(screen,space,camera_offset,truck);truck.draw(screen,camera_offset)
        timer_str=f"Time:{level_time_taken:.2f}" if level_finished else f"Time:{(time.time()-level_start_time):.2f}"
        timer_surf=font.render(timer_str,True,WHITE);screen.blit(timer_surf,(10,10))
        level_surf=font.render(f"Level:{current_level}",True,WHITE);screen.blit(level_surf,(SW-level_surf.get_width()-10,10))
        if level_finished:
            if game_over and current_level==max_level:msg_text="All Levels Complete!";prompt_text="R:Restart Level 1,ESC:Exit"
            else:msg_text="Level Complete!";prompt_text="N:Next Level,R:Retry This Level"
            msg_surf=font.render(msg_text,True,YELLOW);prompt_surf=small_font.render(prompt_text,True,WHITE)
            msg_rect=msg_surf.get_rect(center=(SW//2,SH//2-20));prompt_rect=prompt_surf.get_rect(center=(SW//2,SH//2+20))
            screen.blit(msg_surf,msg_rect);screen.blit(prompt_surf,prompt_rect)
        pygame.display.flip()
    pygame.quit();sys.exit()

if __name__ == "__main__":
    main()