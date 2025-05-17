import numpy as np
import time
import concurrent.futures
import multiprocessing
import pygame

camera_movement_enabled = False  # New global variable

# Global Variables
NUM_CPU_CORES = multiprocessing.cpu_count() - 1
MAX_BOUNCES = 6
focal_length = 800.0

# --- Resolution Control ---
# Set these to your desired render resolution
RENDER_HEIGHT = 325
RENDER_WIDTH = int(RENDER_HEIGHT * 1.5)
# Set these to your desired window/display size (can be different from render res)
WINDOW_HEIGHT = 650
WINDOW_WIDTH = int(WINDOW_HEIGHT * 1.5)
# --------------------------


cam_pos = np.array([150, 97, -800], dtype=np.float32)
cam_yaw = 0.0
cam_pitch = 0.0
move_speed = 200.0
turn_speed = 0.001
accumulated_image = None
accumulation_count = 0
max_accumulation = 1000
movement_threshold = 0.001
frames_still_count = 0
required_still_frames = 2

# Utility Functions
def normalize(v):
    norm = np.sqrt(np.sum(v * v))
    return v / norm if norm > 0 else v

def reflect(v, n):
    return v - 2 * np.dot(v, n) * n

def refract(v, n, ior):
    cos_i = np.clip(np.dot(v, n), -1.0, 1.0)
    if cos_i < 0:
        cos_i = -cos_i
        n = -n
        ior = 1.0 / ior
    sin_t2 = ior * ior * (1.0 - cos_i * cos_i)
    if sin_t2 > 1.0:
        return reflect(v, n)
    cos_t = np.sqrt(1.0 - sin_t2)
    return ior * v + (ior * cos_i - cos_t) * n

def get_camera_vectors(yaw, pitch):
    forward = np.array([np.cos(pitch) * np.sin(yaw), np.sin(pitch), np.cos(pitch) * np.cos(yaw)], dtype=np.float32)
    forward = normalize(forward)
    world_up = np.array([0, 1, 0], dtype=np.float32)
    right = normalize(np.cross(world_up, forward))
    up = normalize(np.cross(forward, right))
    return forward, right, up

# Scene Object Classes
class Material:
    def __init__(self, color, is_glass=False, ior=1.5, reflectivity=0.0):
        self.color = np.array(color, dtype=np.float32)
        self.is_glass = is_glass
        self.ior = ior
        self.reflectivity = reflectivity

class Sphere:
    def __init__(self, center, radius, material):
        self.center = np.array(center, dtype=np.float32)
        self.radius = radius
        self.material = material

class Plane:
    def __init__(self, point, normal, material):
        self.point = np.array(point, dtype=np.float32)
        self.normal = normalize(np.array(normal, dtype=np.float32))
        self.material = material

# Scene Setup
spheres = [
    Sphere([150, 80, 50], 40, Material([1.0, 1.0, 1.0], is_glass=True, ior=1.5, reflectivity=0.1)),
    Sphere([90, 100, 100], 30, Material([1.0, 0.2, 0.2], reflectivity=0.2)),
    Sphere([210, 100, 100], 30, Material([0.2, 0.2, 1.0], reflectivity=0.2)),
    Sphere([60, 60, 80], 20, Material([1.0, 1.0, 0.0], reflectivity=0.9)),
    Sphere([240, 120, 120], 25, Material([0.2, 1.0, 0.2], reflectivity=0.0)),
    Sphere([150, 140, 150], 30, Material([0.6, 0.2, 0.8], reflectivity=0.3)),
    Sphere([150, 40, 80], 30, Material([1.0, 1.0, 1.0], is_glass=True, ior=1.5, reflectivity=0.1)),
    Sphere([220, 40, 120], 30, Material([0.9, 0.9, 0.9], reflectivity=0.3)),
    Sphere([75, 30, 170], 30, Material([0.5, 0.5, 0.5], reflectivity=0.1)),
    Sphere([225, 30, 170], 30, Material([0.7, 0.3, 0.3], reflectivity=0.1)),
]
planes = [Plane(point=[0, 160, 0], normal=[0, -1, 0], material=Material([0.8, 0.8, 0.8], reflectivity=0.0))]
light_pos = np.array([150, 40, -150], dtype=np.float32)

# Ray Tracing Functions
def trace_ray(ray_origin, ray_dir, depth=MAX_BOUNCES):
    if depth <= 0:
        return np.zeros(3)
    closest_t = np.inf
    hit_object = None
    hit_type = None
    for sphere in spheres:
        oc = ray_origin - sphere.center
        a = np.dot(ray_dir, ray_dir)
        b = 2.0 * np.dot(oc, ray_dir)
        c = np.dot(oc, oc) - sphere.radius * sphere.radius
        disc = b * b - 4 * a * c
        if disc > 0:
            t = (-b - np.sqrt(disc)) / (2.0 * a)
            if 0.001 < t < closest_t:
                closest_t = t
                hit_object = sphere
                hit_type = 'sphere'
    for plane in planes:
        denom = np.dot(ray_dir, plane.normal)
        if abs(denom) > 1e-6:
            t = np.dot(plane.point - ray_origin, plane.normal) / denom
            if 0.001 < t < closest_t:
                closest_t = t
                hit_object = plane
                hit_type = 'plane'
    if hit_object is None:
        t = 0.5 * (ray_dir[1] + 1.0)
        return (1.0 - t) * np.array([1.0, 1.0, 1.0]) + t * np.array([0.5, 0.7, 1.0])
    hit_point = ray_origin + ray_dir * closest_t
    if hit_type == 'sphere':
        normal = normalize(hit_point - hit_object.center)
        material = hit_object.material
        if material.is_glass:
            refracted = refract(ray_dir, normal, material.ior)
            new_origin = hit_point + refracted * 0.001
            color = trace_ray(new_origin, refracted, depth - 1)
            return color * material.color * 0.95
        else:
            light_dir = normalize(light_pos - hit_point)
            diffuse = max(np.dot(normal, light_dir), 0.0)
            if material.reflectivity > 0:
                reflected = reflect(ray_dir, normal)
                new_origin = hit_point + reflected * 0.001
                refl_color = trace_ray(new_origin, reflected, depth - 1)
                return refl_color * material.color * material.reflectivity + material.color * diffuse * (1 - material.reflectivity)
            return material.color * diffuse
    else:
        normal = hit_object.normal
        scale = 20.0
        if (int(np.floor(hit_point[0] / scale)) + int(np.floor(hit_point[2] / scale))) % 2 == 0:
            floor_color = np.array([1.0, 1.0, 1.0])
        else:
            floor_color = np.array([0.1, 0.1, 0.1])
        light_dir = normalize(light_pos - hit_point)
        diffuse = max(np.dot(normal, light_dir), 0.0)
        return floor_color * diffuse

def batch_trace_rays(rays_batch):
    return [trace_ray(cam_pos, ray_dir) for ray_dir in rays_batch]

def render_full(cam_pos, cam_yaw, cam_pitch, add_jitter=False):
    forward, right, up = get_camera_vectors(cam_yaw, cam_pitch)
    screen_center = cam_pos + focal_length * forward
    screen_width = 300.0
    screen_height = 200.0
    ys, xs = np.indices((RENDER_HEIGHT, RENDER_WIDTH), dtype=np.float32)
    u = (xs + 0.5) / RENDER_WIDTH
    v = (ys + 0.5) / RENDER_HEIGHT
    if add_jitter:
        u += (np.random.random((RENDER_HEIGHT, RENDER_WIDTH)) - 0.5) / RENDER_WIDTH
        v += (np.random.random((RENDER_HEIGHT, RENDER_WIDTH)) - 0.5) / RENDER_HEIGHT
    offset_x = (u - 0.5) * screen_width
    offset_y = (v - 0.5) * screen_height
    sample_pos = screen_center + offset_x[:, :, np.newaxis] * right + offset_y[:, :, np.newaxis] * up
    ray_dirs = sample_pos - cam_pos
    norms = np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_dirs = ray_dirs / norms
    flat_dirs = ray_dirs.reshape(-1, 3)
    batch_size = max(len(flat_dirs) // (NUM_CPU_CORES * 2), 10)
    batches = [flat_dirs[i:i+batch_size] for i in range(0, len(flat_dirs), batch_size)]
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
        results = list(executor.map(batch_trace_rays, batches))
    flat_result = [item for sublist in results for item in sublist]
    # Ensure result shape matches RENDER_HEIGHT x RENDER_WIDTH
    result = np.array(flat_result, dtype=np.float32).reshape(RENDER_HEIGHT, RENDER_WIDTH, 3) * 255
    return result

# Main Loop
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("First-Person Ray Tracing Game")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)
    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    iteration_future = executor.submit(render_full, cam_pos.copy(), cam_yaw, cam_pitch)
    prev_cam_pos = cam_pos.copy()
    prev_cam_yaw = cam_yaw
    prev_cam_pitch = cam_pitch
    last_time = time.time()
    frame_count = 0
    fps_start_time = time.time()
    running = True
    accumulation_active = False
    accumulation_waiting = False
    loading_font = pygame.font.SysFont(None, 36)
    loading_text = loading_font.render("Initializing Ray Tracer...", True, (255, 255, 255))
    screen.blit(loading_text, (WINDOW_WIDTH//2 - loading_text.get_width()//2, 
                              WINDOW_HEIGHT//2 - loading_text.get_height()//2))
    pygame.display.flip()
    
    while running:
        current_time = time.time()
        delta_time = current_time - last_time
        last_time = current_time

        # Event handling
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_q or event.key == pygame.K_ESCAPE:
                    running = False
                elif event.key == pygame.K_SPACE:
                    camera_movement_enabled = not camera_movement_enabled

        # Update mouse visibility and grab state based on camera_movement_enabled
        if camera_movement_enabled:
            pygame.mouse.set_visible(False)
            pygame.event.set_grab(True)
        else:
            pygame.mouse.set_visible(True)
            pygame.event.set_grab(False)

        # Camera Rotation and Movement
        mouse_dx, mouse_dy = pygame.mouse.get_rel()
        if camera_movement_enabled:
            cam_yaw += mouse_dx * turn_speed
            cam_pitch += mouse_dy * turn_speed
            cam_pitch = np.clip(cam_pitch, -1.5, 1.5)
            keys = pygame.key.get_pressed()
            forward, right, _ = get_camera_vectors(cam_yaw, cam_pitch)
            if keys[pygame.K_w]:
                cam_pos += move_speed * delta_time * forward
            if keys[pygame.K_s]:
                cam_pos -= move_speed * delta_time * forward
            if keys[pygame.K_a]:
                cam_pos -= move_speed * delta_time * right
            if keys[pygame.K_d]:
                cam_pos += move_speed * delta_time * right

        # Check if camera has moved significantly
        pos_diff = np.linalg.norm(cam_pos - prev_cam_pos)
        rot_diff = abs(cam_yaw - prev_cam_yaw) + abs(cam_pitch - prev_cam_pitch)
        camera_moved = (pos_diff > movement_threshold or rot_diff > movement_threshold)

        # Display and Render
        if iteration_future.done():
            new_image = iteration_future.result()
            if camera_moved:
                accumulated_image = new_image.copy()
                accumulation_count = 1
                frames_still_count = 0
                accumulation_active = False
                accumulation_waiting = False
                iteration_future = executor.submit(render_full, cam_pos.copy(), cam_yaw, cam_pitch, False)
            else:
                frames_still_count += 1
                if frames_still_count < required_still_frames:
                    accumulated_image = new_image.copy()
                    accumulation_count = 1
                    accumulation_active = False
                    accumulation_waiting = True
                    iteration_future = executor.submit(render_full, cam_pos.copy(), cam_yaw, cam_pitch, False)
                else:
                    if accumulated_image is None:
                        accumulated_image = new_image.copy()
                        accumulation_count = 1
                    else:
                        weight = min(1.0 / accumulation_count, 0.5)
                        accumulated_image = accumulated_image * (1 - weight) + new_image * weight
                        accumulation_count = min(accumulation_count + 1, max_accumulation)
                    accumulation_active = True
                    accumulation_waiting = False
                    iteration_future = executor.submit(render_full, cam_pos.copy(), cam_yaw, cam_pitch, True)
            
            prev_cam_pos = cam_pos.copy()
            prev_cam_yaw = cam_yaw
            prev_cam_pitch = cam_pitch
            disp_array = np.clip(accumulated_image, 0, 255).astype(np.uint8)
            # Always scale from RENDER_WIDTH/RENDER_HEIGHT to WINDOW_WIDTH/HEIGHT
            lowres_surf = pygame.surfarray.make_surface(disp_array.transpose(1, 0, 2))
            scaled_surf = pygame.transform.scale(lowres_surf, (WINDOW_WIDTH, WINDOW_HEIGHT))
            screen.blit(scaled_surf, (0, 0))

            # FPS and Status Display
            frame_count += 1
            if current_time - fps_start_time >= 1.0:
                fps = frame_count / (current_time - fps_start_time)
                frame_count = 0
                fps_start_time = current_time
                fps_text = font.render(f"FPS: {fps:.2f}", True, (255, 255, 255))
                screen.blit(fps_text, (10, 10))
                status_text = ""
                status_color = (255, 255, 255)
                if accumulation_active:
                    status_text = f"Accumulating: {accumulation_count - 1}/{max_accumulation} samples"
                    status_color = (0, 255, 0)
                elif accumulation_waiting:
                    status_text = f"Waiting: {frames_still_count}/{required_still_frames}"
                    status_color = (255, 255, 0)
                else:
                    status_text = "Camera moving"
                    status_color = (255, 100, 100)
                status_display = font.render(status_text, True, status_color)
                screen.blit(status_display, (10, 40))
                movement_text = font.render(f"Movement: {'Enabled' if camera_movement_enabled else 'Disabled'}", True, (255, 255, 255))
                screen.blit(movement_text, (10, 70))

            pygame.display.flip()

        clock.tick(60)

    pygame.quit()
    executor.shutdown()