import pygame
import pygame.locals as pl
import sys
import math
import random
import numpy as np

# Initialize Pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
FOV = 70
DRAW_DISTANCE = 2600
FAR_DRAW_DISTANCE = 3000
CHUNK_SIZE = 20
TERRAIN_SCALE = 10
SKY_COLOR = (135, 206, 250)
MOVE_SPEED = 2.0
ROTATION_SPEED = 0.03
GRAVITY = 0.03
ENGINE_FORCE = 0.2
MAX_SPEED = 15.0
DRAG_COEFFICIENT = 0.005
CLOUD_COUNT = 80
MOUNTAIN_COUNT = 40
LOD_THRESHOLD = 1000
NEAR_PLANE = 1.0  # Added to improve depth precision

# Set up the display
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("3D Flight Simulator")
clock = pygame.time.Clock()

# Vector3D, Camera, Airplane, Cloud, Mountain classes unchanged
class Vector3D:
    def __init__(self, x=0, y=0, z=0):
        self.x = x
        self.y = y
        self.z = z
    
    def __add__(self, other):
        return Vector3D(self.x + other.x, self.y + other.y, self.z + other.z)
    
    def __sub__(self, other):
        return Vector3D(self.x - other.x, self.y - other.y, self.z - other.z)
    
    def __mul__(self, scalar):
        return Vector3D(self.x * scalar, self.y * scalar, self.z * scalar)
    
    def dot(self, other):
        return self.x * other.x + self.y * other.y + self.z * other.z
    
    def magnitude(self):
        return math.sqrt(self.x**2 + self.y**2 + self.z**2)
    
    def normalize(self):
        mag = self.magnitude()
        if mag == 0:
            return Vector3D(0, 0, 0)
        return Vector3D(self.x / mag, self.y / mag, self.z / mag)
    
    def cross(self, other):
        return Vector3D(
            self.y * other.z - self.z * other.y,
            self.z * other.x - self.x * other.z,
            self.x * other.y - self.y * other.x
        )

class Camera:
    def __init__(self):
        self.position = Vector3D(0, 10, 0)
        self.forward = Vector3D(0, 0, 1)
        self.up = Vector3D(0, 1, 0)
        self.right = Vector3D(1, 0, 0)
        self.pitch = 0
        self.yaw = 0
    
    def update_vectors(self):
        self.forward.x = math.cos(self.yaw) * math.cos(self.pitch)
        self.forward.y = math.sin(self.pitch)
        self.forward.z = math.sin(self.yaw) * math.cos(self.pitch)
        self.forward = self.forward.normalize()
        self.right = self.forward.cross(Vector3D(0, 1, 0)).normalize()
        self.up = self.right.cross(self.forward).normalize()

class Airplane:
    def __init__(self):
        self.position = Vector3D(0, 100, 0)
        self.velocity = Vector3D(0, 0, 0)
        self.acceleration = Vector3D(0, 0, 0)
        self.rotation = Vector3D(0, 0, 0)
        self.forward = Vector3D(0, 0, 1)
        self.up = Vector3D(0, 1, 0)
        self.right = Vector3D(1, 0, 0)
        self.speed = 0
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
        self.acceleration += engine_force
        speed = self.velocity.magnitude()
        if speed > 0.001:
            drag_force = DRAG_COEFFICIENT * speed * speed
            drag_direction = self.velocity * (-1 / speed)
            self.acceleration += drag_direction * drag_force
        self.velocity += self.acceleration
        if self.velocity.magnitude() > MAX_SPEED:
            self.velocity = self.velocity.normalize() * MAX_SPEED
        self.position += self.velocity
        self.acceleration = Vector3D(0, 0, 0)
        if self.position.y < 5:
            self.position.y = 5
            self.velocity.y = max(0, self.velocity.y)

class Cloud:
    def __init__(self):
        self.position = Vector3D(
            random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE),
            random.uniform(100, 400),
            random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE)
        )
        self.size = random.uniform(25, 70)
        self.velocity = Vector3D(random.uniform(-0.5, 0.5), 0, random.uniform(-0.5, 0.5))
    
    def update(self):
        self.position += self.velocity
        wrap_size = FAR_DRAW_DISTANCE
        if self.position.x < -wrap_size:
            self.position.x += 2 * wrap_size
        elif self.position.x > wrap_size:
            self.position.x -= 2 * wrap_size
        if self.position.z < -wrap_size:
            self.position.z += 2 * wrap_size
        elif self.position.z > wrap_size:
            self.position.z -= 2 * wrap_size

class Mountain:
    def __init__(self):
        self.position = Vector3D(
            random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE),
            0,
            random.uniform(-FAR_DRAW_DISTANCE, FAR_DRAW_DISTANCE)
        )
        self.height = random.uniform(50, 250)
        self.radius = random.uniform(50, 140)
        self.color = (
            random.randint(100, 139),
            random.randint(60, 90),
            random.randint(30, 60)
        )

class Ground:
    def __init__(self):
        self.chunk_size = CHUNK_SIZE
        self.scale = TERRAIN_SCALE
        self.chunks = {}
        self.height_cache = {}
    
    def get_height(self, x, z):
        key = (int(x), int(z))
        if key not in self.height_cache:
            height = 0
            for i in range(2):
                freq = 0.01 * (2 ** i)
                amp = 10 / (2 ** i)
                height += amp * math.sin(x * freq + i) * math.cos(z * freq + i)
            self.height_cache[key] = height
        return self.height_cache[key]
    
    def get_terrain_color(self, height):
        if height < -2:
            return (65, 105, 225)
        elif height < 0:
            return (210, 180, 140)
        elif height < 3:
            return (34, 139, 34)
        else:
            return (139, 69, 19)
    
    def generate_chunk(self, chunk_x, chunk_z, lod=1):
        quads = []
        step_size = 5 * lod
        chunk_heights = {}
        # Ensure edge vertices align across chunks
        for x in range(chunk_x * self.chunk_size, (chunk_x + 1) * self.chunk_size + step_size, step_size):
            for z in range(chunk_z * self.chunk_size, (chunk_z + 1) * self.chunk_size + step_size, step_size):
                chunk_heights[(x, z)] = self.get_height(x, z)
        for x in range(chunk_x * self.chunk_size, (chunk_x + 1) * self.chunk_size, step_size):
            for z in range(chunk_z * self.chunk_size, (chunk_z + 1) * self.chunk_size, step_size):
                p1 = (x, chunk_heights[(x, z)], z)
                p2 = (x + step_size, chunk_heights.get((x + step_size, z), self.get_height(x + step_size, z)), z)
                p3 = (x + step_size, chunk_heights.get((x + step_size, z + step_size), self.get_height(x + step_size, z + step_size)), z + step_size)
                p4 = (x, chunk_heights.get((x, z + step_size), self.get_height(x, z + step_size)), z + step_size)
                p1 = (p1[0] * self.scale, p1[1] + 0.01 * (chunk_x + chunk_z), p1[2] * self.scale)
                p2 = (p2[0] * self.scale, p2[1] + 0.01 * (chunk_x + chunk_z), p2[2] * self.scale)
                p3 = (p3[0] * self.scale, p3[1] + 0.01 * (chunk_x + chunk_z), p3[2] * self.scale)
                p4 = (p4[0] * self.scale, p4[1] + 0.01 * (chunk_x + chunk_z), p4[2] * self.scale)
                avg_height = (p1[1] + p2[1] + p3[1] + p4[1]) / 4
                color = self.get_terrain_color(avg_height)
                quads.append((p1, p2, p3, p4, color))
        return quads
    
    def get_height_at_position(self, x, z):
        grid_x = x / self.scale
        grid_z = z / self.scale
        x1 = int(grid_x)
        z1 = int(grid_z)
        x2 = x1 + 1
        z2 = z1 + 1
        h1 = self.get_height(x1, z1)
        h2 = self.get_height(x2, z1)
        h3 = self.get_height(x2, z2)
        h4 = self.get_height(x1, z2)
        fx = grid_x - x1
        fz = grid_z - z1
        h12 = h1 * (1 - fx) + h2 * fx
        h34 = h4 * (1 - fx) + h3 * fx
        return h12 * (1 - fz) + h34 * fz

class Renderer:
    def __init__(self):
        self.width = WIDTH
        self.height = HEIGHT
        self.aspect_ratio = self.width / self.height
        self.fov_rad = math.radians(FOV)
        self.draw_distance = DRAW_DISTANCE
        self.far_draw_distance = FAR_DRAW_DISTANCE
        self.near_plane = NEAR_PLANE
    
    def project_point(self, camera, point):
        point_vec = Vector3D(point[0], point[1], point[2])
        camera_to_point = point_vec - camera.position
        forward_proj = camera_to_point.dot(camera.forward)
        right_proj = camera_to_point.dot(camera.right)
        up_proj = camera_to_point.dot(camera.up)
        if forward_proj <= self.near_plane or forward_proj > self.far_draw_distance:
            return None
        fov_adjustment = math.tan(self.fov_rad / 2)
        x = (right_proj / forward_proj) / fov_adjustment
        y = (up_proj / forward_proj) / fov_adjustment / self.aspect_ratio
        screen_x = int((x + 1) * self.width / 2)
        screen_y = int((1 - y) * self.height / 2)
        return (screen_x, screen_y, forward_proj)
    
    def draw_ground(self, camera, ground, airplane_position):
        projected_quads = []
        chunk_radius = int(self.draw_distance / (ground.chunk_size * ground.scale)) + 2  # Extra buffer
        center_chunk_x = int(airplane_position.x / (ground.chunk_size * ground.scale))
        center_chunk_z = int(airplane_position.z / (ground.chunk_size * ground.scale))
        
        for cx in range(center_chunk_x - chunk_radius, center_chunk_x + chunk_radius + 1):
            for cz in range(center_chunk_z - chunk_radius, center_chunk_z + chunk_radius + 1):
                chunk_center_x = (cx + 0.5) * ground.chunk_size * ground.scale
                chunk_center_z = (cz + 0.5) * ground.chunk_size * ground.scale
                dist = math.sqrt(
                    (chunk_center_x - airplane_position.x) ** 2 +
                    (chunk_center_z - airplane_position.z) ** 2
                )
                if dist > self.far_draw_distance:
                    continue
                chunk_key = (cx, cz)
                lod = 1 if dist < LOD_THRESHOLD else 4
                if chunk_key not in ground.chunks:
                    ground.chunks[chunk_key] = ground.generate_chunk(cx, cz, lod)
                chunk_quads = ground.chunks[chunk_key]
                for i in range(0, len(chunk_quads), 4):
                    for quad in chunk_quads[i:i+4]:
                        center_x = (quad[0][0] + quad[1][0] + quad[2][0] + quad[3][0]) / 4
                        center_y = (quad[0][1] + quad[1][1] + quad[2][1] + quad[3][1]) / 4
                        center_z = (quad[0][2] + quad[1][2] + quad[2][2] + quad[3][2]) / 4
                        projected_center = self.project_point(camera, (center_x, center_y, center_z))
                        if not projected_center:
                            continue
                        p1 = self.project_point(camera, quad[0])
                        p2 = self.project_point(camera, quad[1])
                        p3 = self.project_point(camera, quad[2])
                        p4 = self.project_point(camera, quad[3])
                        if p1 and p2 and p3 and p4:
                            if all(
                                p[0] < -self.width or p[0] >= 2 * self.width or
                                p[1] < -self.height or p[1] >= 2 * self.height
                                for p in [p1, p2, p3, p4]
                            ):
                                continue
                            t = max(0, (projected_center[2] - self.draw_distance) / 
                                    (self.far_draw_distance - self.draw_distance))
                            faded_color = (
                                int(quad[4][0] + t * (SKY_COLOR[0] - quad[4][0])),
                                int(quad[4][1] + t * (SKY_COLOR[1] - quad[4][1])),
                                int(quad[4][2] + t * (SKY_COLOR[2] - quad[4][2]))
                            )
                            projected_quads.append((
                                [p1[0], p1[1]], [p2[0], p2[1]], 
                                [p3[0], p3[1]], [p4[0], p4[1]], 
                                faded_color, projected_center[2]
                            ))
        
        # Unload distant chunks with buffer
        chunks_to_remove = []
        for chunk_key in ground.chunks:
            cx, cz = chunk_key
            dist = math.sqrt(
                ((cx + 0.5) * ground.chunk_size * ground.scale - airplane_position.x) ** 2 +
                ((cz + 0.5) * ground.chunk_size * ground.scale - airplane_position.z) ** 2
            )
            if dist > self.draw_distance + 2 * ground.chunk_size * ground.scale:
                chunks_to_remove.append(chunk_key)
        for chunk_key in chunks_to_remove:
            del ground.chunks[chunk_key]
        
        projected_quads.sort(key=lambda x: -x[5])
        for p1, p2, p3, p4, color, _ in projected_quads:
            pygame.draw.polygon(screen, color, [p1, p2, p3, p4])
    
    def draw_clouds(self, camera, clouds):
        projected_clouds = []
        for cloud in clouds:
            dist_to_camera = math.sqrt(
                (cloud.position.x - camera.position.x) ** 2 +
                (cloud.position.y - camera.position.y) ** 2 +
                (cloud.position.z - camera.position.z) ** 2
            )
            if dist_to_camera > self.far_draw_distance:
                continue
            projected = self.project_point(camera, (cloud.position.x, cloud.position.y, cloud.position.z))
            if projected:
                size = int(cloud.size * 600 / max(1, projected[2]))
                if size > 0:
                    t = max(0, (dist_to_camera - self.draw_distance) / 
                            (self.far_draw_distance - self.draw_distance))
                    color = (
                        int(255 + t * (SKY_COLOR[0] - 255)),
                        int(255 + t * (SKY_COLOR[1] - 255)),
                        int(255 + t * (SKY_COLOR[2] - 255))
                    )
                    projected_clouds.append((projected[0], projected[1], size, dist_to_camera))
        projected_clouds.sort(key=lambda x: -x[3])
        for x, y, size, dist in projected_clouds:
            pygame.draw.circle(screen, color, (x, y), size)
    
    def draw_mountains(self, camera, mountains):
        projected_mountains = []
        for mountain in mountains:
            dist_to_camera = math.sqrt(
                (mountain.position.x - camera.position.x) ** 2 +
                (mountain.position.y - camera.position.y) ** 2 +
                (mountain.position.z - camera.position.z) ** 2
            )
            if dist_to_camera > self.far_draw_distance:
                continue
            projected_base = self.project_point(camera, (mountain.position.x, 0, mountain.position.z))
            if not projected_base:
                continue
            projected_peak = self.project_point(camera, (mountain.position.x, mountain.height, mountain.position.z))
            if projected_peak:
                base_radius = int(mountain.radius * 600 / max(1, projected_base[2]))
                if base_radius > 0:
                    t = max(0, (dist_to_camera - self.draw_distance) / 
                            (self.far_draw_distance - self.draw_distance))
                    color = (
                        int(mountain.color[0] + t * (SKY_COLOR[0] - mountain.color[0])),
                        int(mountain.color[1] + t * (SKY_COLOR[1] - mountain.color[1])),
                        int(mountain.color[2] + t * (SKY_COLOR[2] - mountain.color[2]))
                    )
                    projected_mountains.append((
                        projected_base[0], projected_base[1], 
                        projected_peak[0], projected_peak[1],
                        base_radius, color, dist_to_camera
                    ))
        projected_mountains.sort(key=lambda x: -x[6])
        for base_x, base_y, peak_x, peak_y, base_radius, color, _ in projected_mountains:
            pygame.draw.polygon(screen, color, [
                (base_x - base_radius, base_y),
                (peak_x, peak_y),
                (base_x + base_radius, base_y)
            ])
    
    def draw_airplane(self, camera, airplane):
        nose = airplane.position + airplane.forward * 10
        tail = airplane.position - airplane.forward * 10
        left_wing_tip = airplane.position - airplane.right * 15
        right_wing_tip = airplane.position + airplane.right * 15
        left_stabilizer = tail - airplane.right * 5
        right_stabilizer = tail + airplane.right * 5
        vertical_stabilizer = tail + airplane.up * 5
        proj_nose = self.project_point(camera, (nose.x, nose.y, nose.z))
        proj_tail = self.project_point(camera, (tail.x, tail.y, tail.z))
        proj_left_wing = self.project_point(camera, (left_wing_tip.x, left_wing_tip.y, left_wing_tip.z))
        proj_right_wing = self.project_point(camera, (right_wing_tip.x, right_wing_tip.y, right_wing_tip.z))
        proj_left_stab = self.project_point(camera, (left_stabilizer.x, left_stabilizer.y, left_stabilizer.z))
        proj_right_stab = self.project_point(camera, (right_stabilizer.x, right_stabilizer.y, right_stabilizer.z))
        proj_vert_stab = self.project_point(camera, (vertical_stabilizer.x, vertical_stabilizer.y, vertical_stabilizer.z))
        if proj_nose and proj_tail:
            pygame.draw.line(screen, (220, 220, 220), (proj_nose[0], proj_nose[1]), (proj_tail[0], proj_tail[1]), 4)
            if proj_left_wing and proj_right_wing:
                pygame.draw.line(screen, (200, 200, 200), (proj_left_wing[0], proj_left_wing[1]), 
                                (proj_right_wing[0], proj_right_wing[1]), 3)
                pygame.draw.line(screen, (200, 200, 200), (proj_nose[0], proj_nose[1]), 
                                (proj_left_wing[0], proj_left_wing[1]), 2)
                pygame.draw.line(screen, (200, 200, 200), (proj_nose[0], proj_nose[1]), 
                                (proj_right_wing[0], proj_right_wing[1]), 2)
            if proj_left_stab and proj_right_stab and proj_vert_stab:
                pygame.draw.line(screen, (180, 180, 180), (proj_left_stab[0], proj_left_stab[1]), 
                                (proj_right_stab[0], proj_right_stab[1]), 2)
                pygame.draw.line(screen, (180, 180, 180), (proj_tail[0], proj_tail[1]), 
                                (proj_vert_stab[0], proj_vert_stab[1]), 2)
    
    def draw_hud(self, airplane):
        altitude_text = f"Altitude: {int(airplane.position.y)} m"
        font = pygame.font.SysFont(None, 24)
        altitude_surface = font.render(altitude_text, True, (255, 255, 255))
        screen.blit(altitude_surface, (10, 10))
        speed_text = f"Speed: {int(airplane.velocity.magnitude() * 50)} km/h"
        speed_surface = font.render(speed_text, True, (255, 255, 255))
        screen.blit(speed_surface, (10, 40))
        pygame.draw.rect(screen, (50, 50, 50), pygame.Rect(10, 70, 100, 20))
        pygame.draw.rect(screen, (255, 0, 0), pygame.Rect(10, 70, 100 * airplane.throttle, 20))
        throttle_text = f"Throttle: {int(airplane.throttle * 100)}%"
        throttle_surface = font.render(throttle_text, True, (255, 255, 255))
        screen.blit(throttle_surface, (120, 70))
        horizon_center = (WIDTH - 100, 100)
        pygame.draw.circle(screen, (0, 0, 0), horizon_center, 50, 2)
        pitch_offset = math.sin(airplane.rotation.x) * 40
        roll_angle = -airplane.rotation.z
        pygame.draw.line(
            screen,
            (150, 150, 150),
            (
                horizon_center[0] - 50 * math.cos(roll_angle),
                horizon_center[1] - 50 * math.sin(roll_angle) + pitch_offset
            ),
            (
                horizon_center[0] + 50 * math.cos(roll_angle),
                horizon_center[1] + 50 * math.sin(roll_angle) + pitch_offset
            ),
            2
        )
        camera_mode_text = f"Camera: {'First Person' if camera_mode == 'first_person' else 'Third Person'}"
        camera_surface = font.render(camera_mode_text, True, (255, 255, 255))
        screen.blit(camera_surface, (WIDTH - 200, HEIGHT - 30))
        controls_text = "Controls: Arrow keys - Pitch/Yaw | W/S - Throttle | A/D - Roll | C - Toggle Camera"
        controls_surface = font.render(controls_text, True, (255, 255, 255))
        screen.blit(controls_surface, (10, HEIGHT - 30))

# Initialize game objects
airplane = Airplane()
camera = Camera()
renderer = Renderer()
ground = Ground()
clouds = [Cloud() for _ in range(CLOUD_COUNT)]
mountains = [Mountain() for _ in range(MOUNTAIN_COUNT)]
camera_mode = "third_person"
camera_distance = 50
camera_height = 15

# Main game loop
running = True
last_time = pygame.time.get_ticks()
fps_counter = 0
fps_timer = 0
fps = 0

while running:
    current_time = pygame.time.get_ticks()
    dt = (current_time - last_time) / 1000.0
    last_time = current_time
    
    fps_counter += 1
    fps_timer += dt
    if fps_timer >= 1.0:
        fps = fps_counter
        fps_counter = 0
        fps_timer = 0
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_ESCAPE:
                running = False
            elif event.key == pygame.K_c:
                camera_mode = "first_person" if camera_mode == "third_person" else "third_person"
    
    keys = pygame.key.get_pressed()
    if keys[pygame.K_w]:
        airplane.throttle = min(1.0, airplane.throttle + 0.01)
    if keys[pygame.K_s]:
        airplane.throttle = max(0.0, airplane.throttle - 0.01)
    if keys[pygame.K_UP]:
        airplane.rotation.x += ROTATION_SPEED
    if keys[pygame.K_DOWN]:
        airplane.rotation.x -= ROTATION_SPEED
    if keys[pygame.K_LEFT]:
        airplane.rotation.y -= ROTATION_SPEED
    if keys[pygame.K_RIGHT]:
        airplane.rotation.y += ROTATION_SPEED
    if keys[pygame.K_a]:
        airplane.rotation.z += ROTATION_SPEED
    if keys[pygame.K_d]:
        airplane.rotation.z -= ROTATION_SPEED
    
    airplane.rotation.x = max(min(airplane.rotation.x, math.pi/4), -math.pi/4)
    airplane.rotation.z = max(min(airplane.rotation.z, math.pi/4), -math.pi/4)
    
    airplane.update_vectors()
    airplane.apply_physics()
    
    terrain_height = ground.get_height_at_position(airplane.position.x, airplane.position.z)
    if airplane.position.y < terrain_height + 5:
        airplane.position.y = terrain_height + 5
        airplane.velocity.y = max(0, airplane.velocity.y)
    
    for cloud in clouds:
        cloud.update()
    
    if camera_mode == "third_person":
        target_offset = airplane.forward * -camera_distance
        target_offset.y += camera_height
        target_pos = airplane.position + target_offset
        t = 0.1
        camera.position.x += (target_pos.x - camera.position.x) * t
        camera.position.y += (target_pos.y - camera.position.y) * t
        camera.position.z += (target_pos.z - camera.position.z) * t
        look_direction = airplane.position - camera.position
        camera.pitch = math.asin(look_direction.y / look_direction.magnitude())
        camera.yaw = math.atan2(look_direction.z, look_direction.x)
    else:
        camera.position = airplane.position
        camera.pitch = airplane.rotation.x
        camera.yaw = airplane.rotation.y
    
    camera.update_vectors()
    
    screen.fill(SKY_COLOR)
    
    renderer.draw_mountains(camera, mountains)
    renderer.draw_ground(camera, ground, airplane.position)
    renderer.draw_clouds(camera, clouds)
    
    if camera_mode == "third_person":
        renderer.draw_airplane(camera, airplane)
    
    renderer.draw_hud(airplane)
    
    fps_text = f"FPS: {fps}"
    fps_surface = pygame.font.SysFont(None, 24).render(fps_text, True, (255, 255, 255))
    screen.blit(fps_surface, (WIDTH - 80, 10))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
sys.exit()