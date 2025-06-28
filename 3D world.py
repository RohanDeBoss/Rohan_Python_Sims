import pygame
import pygame.locals as pl
import sys
import math
import random
import numpy as np

# Initialize Pygame
pygame.init()

# --- Main Constants ---
WIDTH, HEIGHT = 900, 600
FOV = 70
DRAW_DISTANCE = 2500
FAR_DRAW_DISTANCE = 3100
CHUNK_SIZE = 17
TERRAIN_SCALE = 10
SKY_COLOR = (135, 206, 250)
NEAR_PLANE = 1.0
CLOUD_COUNT = 80
MOUNTAIN_COUNT = 40
LOD_THRESHOLD = 1000

# --- Camera & Flight Model Constants (Easy to Tweak) ---
PITCH_SPEED = 0.03
YAW_SPEED = 0.02
MAX_PITCH_ANGLE = 1.5

GRAVITY = 0.03
ENGINE_FORCE = 0.17
MAX_SPEED = 13.0
DRAG_COEFFICIENT = 0.005

CAMERA_DISTANCE = 60
CAMERA_HEIGHT = 45
CAMERA_LOOK_DOWN_OFFSET = 10
# FIX: New constant to control the zoom intensity.
# 0.0 = No zoom at all. 1.0 = Original, full zoom.
CAMERA_ZOOM_EFFECT_STRENGTH = 0.3 

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Flight Simulator")
clock = pygame.time.Clock()

# --- Classes are identical to the stable version ---
class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x, self.y, self.z = x, y, z
    def __add__(self, other): return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    def __sub__(self, other): return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    def __mul__(self, scalar): return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    def dot(self, other): return self.x * other.x + self.y * other.y + self.z * other.z
    def magnitude(self): return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    def normalize(self):
        mag = self.magnitude()
        return Vector3D(self.x / mag, self.y / mag, self.z / mag) if mag != 0 else Vector3D(0, 0, 0)
    def cross(self, other): return Vector3D(self.y * other.z - self.z * other.y, self.z * other.x - self.x * other.z, self.x * other.y - self.y * other.x)

class Camera:
    def __init__(self):
        self.position, self.forward, self.up, self.right = Vector3D(0, 10, 0), Vector3D(0, 0, 1), Vector3D(0, 1, 0), Vector3D(1, 0, 0)
        self.pitch, self.yaw = 0, 0
    def update_vectors(self):
        self.forward.x = math.cos(self.yaw) * math.cos(self.pitch)
        self.forward.y = math.sin(self.pitch)
        self.forward.z = math.sin(self.yaw) * math.cos(self.pitch)
        self.forward = self.forward.normalize()
        self.right = self.forward.cross(Vector3D(0, 1, 0)).normalize()
        self.up = self.right.cross(self.forward).normalize()

class Airplane:
    def __init__(self):
        self.position, self.velocity, self.acceleration = Vector3D(0, 100, 0), Vector3D(0, 0, 0), Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.forward, self.up, self.right = Vector3D(0, 0, 1), Vector3D(0, 1, 0), Vector3D(1, 0, 0)
        self.throttle = 0.5
    def update_vectors(self):
        self.forward.x = math.cos(self.rotation.y) * math.cos(self.rotation.x)
        self.forward.y = math.sin(self.rotation.x)
        self.forward.z = math.sin(self.rotation.y) * math.cos(self.rotation.x)
        self.forward = self.forward.normalize()
        self.right = self.forward.cross(Vector3D(0, 1, 0)).normalize()
        self.up = self.right.cross(self.forward).normalize()
    def apply_physics(self):
        self.acceleration.y -= GRAVITY
        engine_force = self.forward * (self.throttle * ENGINE_FORCE)
        self.acceleration = self.acceleration + engine_force
        speed = self.velocity.magnitude()
        if speed > 0.001:
            drag_force = DRAG_COEFFICIENT * speed * speed
            drag_direction = self.velocity * (-1 / speed)
            self.acceleration = self.acceleration + drag_direction * drag_force
        self.velocity = self.velocity + self.acceleration
        if self.velocity.magnitude() > MAX_SPEED: self.velocity = self.velocity.normalize() * MAX_SPEED
        self.position = self.position + self.velocity
        self.acceleration = Vector3D(0, 0, 0)
        if self.position.y < 5: self.position.y, self.velocity.y = 5, max(0, self.velocity.y)

class Cloud:
    def __init__(self):
        self.position = Vector3D(random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE), random.uniform(100, 400), random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE))
        self.size, self.velocity = random.uniform(25, 70), Vector3D(random.uniform(-0.5, 0.5), 0, random.uniform(-0.5, 0.5))
    def update(self):
        self.position = self.position + self.velocity
        for axis in ['x', 'z']:
            pos = getattr(self.position, axis)
            if pos < -FAR_DRAW_DISTANCE: setattr(self.position, axis, pos + 2 * FAR_DRAW_DISTANCE)
            elif pos > FAR_DRAW_DISTANCE: setattr(self.position, axis, pos - 2 * FAR_DRAW_DISTANCE)

class Mountain:
    def __init__(self):
        self.position = Vector3D(random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE), 0, random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE))
        self.height, self.radius = random.uniform(50, 250), random.uniform(50, 140)
        self.color = (random.randint(100, 139), random.randint(60, 90), random.randint(30, 60))

class Ground:
    def __init__(self):
        self.chunk_size, self.scale, self.chunks, self.height_cache = CHUNK_SIZE, TERRAIN_SCALE, {}, {}
    def get_height(self, x, z):
        key = (int(x), int(z))
        if key not in self.height_cache: self.height_cache[key] = sum(10 / (2**i) * math.sin(x*0.01*(2**i)+i)*math.cos(z*0.01*(2**i)+i) for i in range(2))
        return self.height_cache[key]
    def get_terrain_color(self, h):
        if h < -2: return (65, 105, 225)
        elif h < 0: return (210, 180, 140)
        elif h < 3: return (34, 139, 34)
        else: return (139, 69, 19)
    def generate_chunk(self, chunk_x, chunk_z, lod=1):
        quads, step_size, chunk_heights = [], 5 * lod, {}
        for x in range(chunk_x * CHUNK_SIZE, (chunk_x+1) * CHUNK_SIZE + step_size, step_size):
            for z in range(chunk_z * CHUNK_SIZE, (chunk_z+1) * CHUNK_SIZE + step_size, step_size):
                chunk_heights[(x, z)] = self.get_height(x, z)
        for x in range(chunk_x * CHUNK_SIZE, (chunk_x+1) * CHUNK_SIZE, step_size):
            for z in range(chunk_z * CHUNK_SIZE, (chunk_z+1) * CHUNK_SIZE, step_size):
                p1 = (x*self.scale, chunk_heights[(x,z)], z*self.scale)
                p2 = ((x+step_size)*self.scale, chunk_heights.get((x+step_size, z), self.get_height(x+step_size,z)), z*self.scale)
                p3 = ((x+step_size)*self.scale, chunk_heights.get((x+step_size, z+step_size),self.get_height(x+step_size,z+step_size)), (z+step_size)*self.scale)
                p4 = (x*self.scale, chunk_heights.get((x, z+step_size),self.get_height(x,z+step_size)), (z+step_size)*self.scale)
                avg_h = (p1[1]+p2[1]+p3[1]+p4[1])/4
                quads.append((p1, p2, p3, p4, self.get_terrain_color(avg_h)))
        return quads
    def get_height_at_position(self, x, z):
        gx, gz = x / self.scale, z / self.scale
        x1, z1 = int(gx), int(gz)
        h1,h2,h3,h4 = self.get_height(x1,z1), self.get_height(x1+1,z1), self.get_height(x1+1,z1+1), self.get_height(x1,z1+1)
        fx, fz = gx - x1, gz - z1
        return (h1*(1-fx)+h2*fx)*(1-fz) + (h4*(1-fx)+h3*fx)*fz

class Renderer:
    def __init__(self):
        self.width, self.height, self.aspect_ratio = WIDTH, HEIGHT, WIDTH / HEIGHT
        self.fov_rad = math.radians(FOV)
        self.draw_distance, self.far_draw_distance, self.near_plane = DRAW_DISTANCE, FAR_DRAW_DISTANCE, NEAR_PLANE
        try:
            self.font_main = pygame.font.SysFont('Arial', 20)
            self.font_small = pygame.font.SysFont('Arial', 14)
        except pygame.error:
            self.font_main = pygame.font.SysFont(None, 24)
            self.font_small = pygame.font.SysFont(None, 18)

    def project_point(self, camera, point):
        p_vec, cam_to_p = Vector3D(*point), Vector3D(*point) - camera.position
        f_proj = cam_to_p.dot(camera.forward)
        if not (self.near_plane < f_proj < self.far_draw_distance): return None
        r_proj, u_proj = cam_to_p.dot(camera.right), cam_to_p.dot(camera.up)
        fov_adj = math.tan(self.fov_rad / 2)
        x, y = (r_proj / f_proj) / fov_adj, (u_proj / f_proj) / fov_adj / self.aspect_ratio
        return (int((x + 1) * self.width / 2), int((1 - y) * self.height / 2), f_proj)

    def draw_ground(self, camera, ground, airplane_position):
        quads, radius = [], int(self.draw_distance / (ground.chunk_size * ground.scale)) + 2
        cx_c, cz_c = int(airplane_position.x / (ground.chunk_size * ground.scale)), int(airplane_position.z / (ground.chunk_size * ground.scale))
        for cx in range(cx_c - radius, cx_c + radius + 1):
            for cz in range(cz_c - radius, cz_c + radius + 1):
                chunk_center = Vector3D((cx + 0.5) * ground.chunk_size * ground.scale, 0, (cz + 0.5) * ground.chunk_size * ground.scale)
                if (chunk_center - airplane_position).magnitude() > self.far_draw_distance: continue
                key = (cx, cz)
                lod = 1 if (chunk_center - airplane_position).magnitude() < LOD_THRESHOLD else 4
                if key not in ground.chunks: ground.chunks[key] = ground.generate_chunk(cx, cz, lod)
                for quad in ground.chunks[key]:
                    projs = [self.project_point(camera, p) for p in quad[:4]]
                    if all(projs):
                        depth = sum(p[2] for p in projs) / 4
                        if all(p[0] < -self.width or p[0] >= 2 * self.width or p[1] < -self.height or p[1] >= 2 * self.height for p in projs): continue
                        t = max(0, (depth - self.draw_distance) / (self.far_draw_distance - self.draw_distance))
                        color = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(quad[4], SKY_COLOR))
                        quads.append(([p[:2] for p in projs], color, depth))
        chunks_to_remove = [k for k, v in ground.chunks.items() if (Vector3D((k[0] + 0.5) * ground.chunk_size * ground.scale, 0, (k[1] + 0.5) * ground.chunk_size * ground.scale) - airplane_position).magnitude() > self.draw_distance + 2 * ground.chunk_size * ground.scale]
        for k in chunks_to_remove: del ground.chunks[k]
        quads.sort(key=lambda x: -x[2])
        for points, color, _ in quads: pygame.draw.polygon(screen, color, points)
    
    def draw_clouds(self, camera, clouds):
        projected = sorted([ (c, self.project_point(camera, (c.position.x, c.position.y, c.position.z))) for c in clouds if (c.position - camera.position).magnitude() <= self.far_draw_distance ], key=lambda x: -x[1][2] if x[1] else 0)
        for cloud, proj in projected:
            if not proj: continue
            size = int(cloud.size * 600 / max(1, proj[2]))
            if size > 0:
                t = max(0, (proj[2] - self.draw_distance) / (self.far_draw_distance - self.draw_distance))
                color = tuple(int(255 + t * (c - 255)) for c in SKY_COLOR)
                pygame.draw.circle(screen, color, proj[:2], size)

    def draw_mountains(self, camera, mountains):
        projected = sorted([(m, (m.position - camera.position).magnitude()) for m in mountains if (m.position - camera.position).magnitude() <= self.far_draw_distance], key=lambda x: -x[1])
        for mountain, dist in projected:
            peak = self.project_point(camera, (mountain.position.x, mountain.height, mountain.position.z))
            if peak:
                base = self.project_point(camera, (mountain.position.x, 0, mountain.position.z))
                if base:
                    radius = int(mountain.radius * 600 / max(1, base[2]))
                    if radius > 0:
                        t = max(0, (dist - self.draw_distance) / (self.far_draw_distance - self.draw_distance))
                        color = tuple(int(c1 + t * (c2 - c1)) for c1, c2 in zip(mountain.color, SKY_COLOR))
                        pygame.draw.polygon(screen, color, [(base[0] - radius, base[1]), peak[:2], (base[0] + radius, base[1])])

    def draw_airplane(self, camera, airplane):
        C_TOP, C_SIDE, C_BOTTOM = (220, 225, 230), (180, 185, 190), (140, 145, 150)
        C_ACCENT, C_OUTLINE = (200, 50, 50), (50, 50, 55)
        pos, fwd, up, rgt = airplane.position, airplane.forward, airplane.up, airplane.right
        v = {
            "nose": pos + fwd * 12, "cockpit_peak": pos + fwd * 5 + up * 2.5,
            "fuselage_top_rear": pos - fwd * 13 + up * 1, "fuselage_bottom_rear": pos - fwd * 13 - up * 1.5,
            "fuselage_bottom_mid": pos + fwd * 2 - up * 1.5, "wing_L_root_front": pos + fwd * 4 - rgt * 1,
            "wing_L_root_rear": pos - fwd * 4 - rgt * 1, "wing_L_tip": pos - fwd * 4 - rgt * 16,
            "wing_R_root_front": pos + fwd * 4 + rgt * 1, "wing_R_root_rear": pos - fwd * 4 + rgt * 1,
            "wing_R_tip": pos - fwd * 4 + rgt * 16, "v_stab_peak": pos - fwd * 11 + up * 5,
            "h_stab_L": pos - fwd * 12 - rgt * 5, "h_stab_R": pos - fwd * 12 + rgt * 5,
        }
        polygons = [
            (C_SIDE, [v["nose"], v["fuselage_bottom_mid"], v["wing_L_root_front"], v["cockpit_peak"]]),
            (C_SIDE, [v["nose"], v["cockpit_peak"], v["wing_R_root_front"], v["fuselage_bottom_mid"]]),
            (C_SIDE, [v["wing_L_root_front"], v["wing_L_root_rear"], v["fuselage_top_rear"], v["cockpit_peak"]]),
            (C_SIDE, [v["wing_R_root_front"], v["cockpit_peak"], v["fuselage_top_rear"], v["wing_R_root_rear"]]),
            (C_BOTTOM, [v["fuselage_bottom_mid"], v["fuselage_bottom_rear"], v["wing_L_root_rear"], v["wing_L_root_front"]]),
            (C_BOTTOM, [v["fuselage_bottom_mid"], v["wing_R_root_front"], v["wing_R_root_rear"], v["fuselage_bottom_rear"]]),
            (C_ACCENT, [v["v_stab_peak"], v["fuselage_top_rear"], v["fuselage_bottom_rear"]]),
            (C_TOP, [v["h_stab_L"], v["fuselage_top_rear"], v["h_stab_R"]]),
            (C_TOP, [v["wing_L_root_front"], v["wing_L_root_rear"], v["wing_L_tip"]]),
            (C_TOP, [v["wing_R_root_front"], v["wing_R_tip"], v["wing_R_root_rear"]]),
            (C_TOP, [v["cockpit_peak"], v["fuselage_top_rear"], v["nose"]]),
        ]
        projected_polygons = []
        for color, vertices in polygons:
            points_2d = [self.project_point(camera, (vert.x, vert.y, vert.z)) for vert in vertices]
            if all(points_2d):
                avg_depth = sum(p[2] for p in points_2d) / len(points_2d)
                screen_points = [p[:2] for p in points_2d]
                projected_polygons.append((screen_points, color, avg_depth))
        projected_polygons.sort(key=lambda x: -x[2])
        for points, color, _ in projected_polygons:
            pygame.draw.polygon(screen, color, points)
            pygame.draw.polygon(screen, C_OUTLINE, points, 1)
    
    def draw_hud(self, airplane):
        self.draw_info_panel(airplane); self.draw_attitude_indicator(airplane); self.draw_camera_mode()
    def draw_info_panel(self, airplane):
        panel_surf = pygame.Surface((220, 90), pygame.SRCALPHA); panel_surf.fill((0, 0, 0, 120))
        alt_text = self.font_main.render(f"ALT {int(airplane.position.y):>4} m", True, (255, 255, 255))
        spd_text = self.font_main.render(f"SPD {int(airplane.velocity.magnitude() * 50):>3} km/h", True, (255, 255, 255))
        panel_surf.blit(alt_text, (10, 10)); panel_surf.blit(spd_text, (10, 35))
        thr_text = self.font_small.render(f"THR {int(airplane.throttle * 100)}%", True, (255, 255, 255))
        pygame.draw.rect(panel_surf, (80, 80, 80), (10, 65, 150, 15))
        pygame.draw.rect(panel_surf, (255, 128, 0), (10, 65, 150 * airplane.throttle, 15))
        panel_surf.blit(thr_text, (165, 65)); screen.blit(panel_surf, (10, 10))
    def draw_attitude_indicator(self, airplane):
        center, radius = (WIDTH - 100, 100), 60
        indicator_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        clip_surf = pygame.Surface((radius * 2, radius * 2), pygame.SRCALPHA)
        pygame.draw.circle(clip_surf, (255, 255, 255), (radius, radius), radius)
        pitch_offset = math.sin(airplane.rotation.x) * radius * 1.5
        pygame.draw.rect(indicator_surf, (101, 67, 33), (0, radius + pitch_offset, radius * 2, radius * 2))
        pygame.draw.rect(indicator_surf, (80, 150, 255), (0, -radius, radius * 2, radius * 2 + pitch_offset))
        indicator_surf.blit(clip_surf, (0, 0), special_flags=pygame.BLEND_RGBA_MULT)
        screen.blit(indicator_surf, (center[0] - radius, center[1] - radius))
        pygame.draw.circle(screen, (0, 0, 0, 150), center, radius, 3)
        pygame.draw.line(screen, (255, 255, 255), (center[0] - 20, center[1]), (center[0] - 10, center[1]), 2)
        pygame.draw.line(screen, (255, 255, 255), (center[0] + 10, center[1]), (center[0] + 20, center[1]), 2)
        pygame.draw.line(screen, (255, 255, 255), (center[0], center[1] - 5), (center[0], center[1] + 5), 2)
    def draw_camera_mode(self):
        panel_surf = pygame.Surface((180, 30), pygame.SRCALPHA); panel_surf.fill((0, 0, 0, 120))
        cam_text = self.font_small.render(f"C: Toggle Camera ({camera_mode.replace('_', ' ').title()})", True, (255, 255, 255))
        text_rect = cam_text.get_rect(center=(panel_surf.get_width()/2, panel_surf.get_height()/2))
        panel_surf.blit(cam_text, text_rect); screen.blit(panel_surf, (WIDTH - 190, HEIGHT - 40))

# --- Main Loop ---
airplane, camera, renderer, ground = Airplane(), Camera(), Renderer(), Ground()
clouds, mountains = [Cloud() for _ in range(CLOUD_COUNT)], [Mountain() for _ in range(MOUNTAIN_COUNT)]
camera_mode = "third_person"

running, last_time, fps_counter, fps_timer, fps = True, pygame.time.get_ticks(), 0, 0, 0
while running:
    current_time, dt = pygame.time.get_ticks(), (pygame.time.get_ticks() - last_time) / 1000.0; last_time = current_time
    fps_counter += 1; fps_timer += dt
    if fps_timer >= 1.0: fps, fps_counter, fps_timer = fps_counter, 0, 0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT or (event.type == pygame.KEYDOWN and event.key == pygame.K_ESCAPE): running = False
        elif event.type == pygame.KEYDOWN and event.key == pygame.K_c: camera_mode = "first_person" if camera_mode == "third_person" else "third_person"
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]: airplane.throttle = min(1.0, airplane.throttle + 0.01)
    if keys[pygame.K_s]: airplane.throttle = max(0.0, airplane.throttle - 0.01)
    if keys[pygame.K_UP]: airplane.rotation.x += PITCH_SPEED
    if keys[pygame.K_DOWN]: airplane.rotation.x -= PITCH_SPEED
    if keys[pygame.K_LEFT]: airplane.rotation.y -= YAW_SPEED
    if keys[pygame.K_RIGHT]: airplane.rotation.y += YAW_SPEED
    airplane.rotation.x = max(-MAX_PITCH_ANGLE, min(airplane.rotation.x, MAX_PITCH_ANGLE))
    
    airplane.update_vectors(); airplane.apply_physics()
    
    terrain_height = ground.get_height_at_position(airplane.position.x, airplane.position.z)
    if airplane.position.y < terrain_height + 5: airplane.position.y, airplane.velocity.y = terrain_height + 5, max(0, airplane.velocity.y)
    
    for cloud in clouds: cloud.update()
    
    if camera_mode == "third_person":
        # FIX: Blend between a horizontal vector and the plane's actual forward vector
        # This creates a dampened zoom effect that is less exaggerated.
        horizontal_forward = Vector3D(math.cos(airplane.rotation.y), 0, math.sin(airplane.rotation.y)).normalize()
        
        # Linear interpolation (lerp) between the two vectors
        strength = CAMERA_ZOOM_EFFECT_STRENGTH
        blended_forward = horizontal_forward * (1 - strength) + airplane.forward * strength
        
        target_offset = blended_forward * -CAMERA_DISTANCE
        target_offset.y += CAMERA_HEIGHT
        
        target_pos = airplane.position + target_offset
        t = 0.1
        camera.position.x += (target_pos.x - camera.position.x) * t
        camera.position.y += (target_pos.y - camera.position.y) * t
        camera.position.z += (target_pos.z - camera.position.z) * t
        
        camera_look_at_target = airplane.position - Vector3D(0, CAMERA_LOOK_DOWN_OFFSET, 0)
        look_direction = camera_look_at_target - camera.position
        
        try:
            camera.pitch, camera.yaw = math.asin(look_direction.y / look_direction.magnitude()), math.atan2(look_direction.z, look_direction.x)
        except (ValueError, ZeroDivisionError): pass
    else: camera.position, camera.pitch, camera.yaw = airplane.position, airplane.rotation.x, airplane.rotation.y
    camera.update_vectors()
    
    screen.fill(SKY_COLOR)
    renderer.draw_mountains(camera, mountains)
    renderer.draw_ground(camera, ground, airplane.position)
    renderer.draw_clouds(camera, clouds)
    if camera_mode == "third_person": renderer.draw_airplane(camera, airplane)
    
    renderer.draw_hud(airplane)
    
    fps_surface = renderer.font_small.render(f"FPS: {fps}", True, (255, 255, 255))
    screen.blit(fps_surface, (WIDTH - 50, 10))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()