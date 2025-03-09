import numpy as np
import pygame
import time
import concurrent.futures
import multiprocessing
import functools

# ---------------- Global Variables ----------------
NUM_CPU_CORES = multiprocessing.cpu_count() - 1  # Number of CPU cores to use

# ---------------- Parameters ----------------
MAX_BOUNCES = 10  # Maximum recursion depth (number of light bounces)

# ---------------- Resolution & Camera Settings ----------------
WINDOW_HEIGHT = 600   # Display window height
WINDOW_WIDTH = int(WINDOW_HEIGHT * 1.5)   # Display window width
RENDER_HEIGHT = 300   # Render height
RENDER_WIDTH = int(RENDER_HEIGHT * 1.5)    # Render width
CAMERA_POS = np.array([150, 97, -800], dtype=np.float32)  # Camera position

# ---------------- Utility Functions ----------------
def normalize(v):
    """Normalize a vector."""
    norm = np.sqrt(np.sum(v * v))
    return v / norm if norm > 0 else v

def reflect(v, n):
    """Compute the reflection of vector v around normal n."""
    return v - 2 * np.dot(v, n) * n

def refract(v, n, ior):
    """Compute refraction of vector v through a surface with normal n and index of refraction ior."""
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

# ---------------- Scene Object Classes ----------------
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

# ---------------- Scene Setup ----------------
spheres = [
    Sphere([300 * 0.5, 200 * 0.4, 50], 40,
           Material([1.0, 1.0, 1.0], is_glass=True, ior=1.5, reflectivity=0.1)),
    Sphere([300 * 0.3, 200 * 0.5, 100], 30,
           Material([1.0, 0.2, 0.2], reflectivity=0.2)),
    Sphere([300 * 0.7, 200 * 0.5, 100], 30,
           Material([0.2, 0.2, 1.0], reflectivity=0.2)),
    Sphere([300 * 0.2, 200 * 0.3, 80], 20,
           Material([1.0, 1.0, 0.0], reflectivity=0.9)),
    Sphere([300 * 0.8, 200 * 0.6, 120], 25,
           Material([0.2, 1.0, 0.2], reflectivity=0.0)),
    Sphere([300 * 0.5, 200 * 0.7, 150], 30,
           Material([0.6, 0.2, 0.8], reflectivity=0.3)),
    Sphere([150, 40, 80], 30, Material([1.0, 1.0, 1.0], is_glass=True, ior=1.5, reflectivity=0.1)),
    Sphere([220, 40, 120], 30, Material([0.9, 0.9, 0.9], reflectivity=0.3)),
    Sphere([75, 30, 170], 30, Material([0.5, 0.5, 0.5], reflectivity=0.1)),
    Sphere([225, 30, 170], 30, Material([0.7, 0.3, 0.3], reflectivity=0.1)),
]

planes = [
    Plane(point=[0, 160, 0], normal=[0, -1, 0],
          material=Material([0.8, 0.8, 0.8], reflectivity=0.0))
]

light_pos = np.array([300 * 0.5, 200 * 0.2, -150], dtype=np.float32)  # Single light source

# ---------------- Ray Tracing Function ----------------
def trace_ray(ray_origin, ray_dir, depth=MAX_BOUNCES):
    """Trace a ray through the scene and compute the resulting color."""
    if depth <= 0:
        return np.zeros(3)
    
    closest_t = np.inf
    hit_object = None
    hit_type = None  # 'sphere' or 'plane'
    
    # Sphere intersections
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
    
    # Plane intersections
    for plane in planes:
        denom = np.dot(ray_dir, plane.normal)
        if abs(denom) > 1e-6:
            t = np.dot(plane.point - ray_origin, plane.normal) / denom
            if 0.001 < t < closest_t:
                closest_t = t
                hit_object = plane
                hit_type = 'plane'
    
    # No hit: return sky color
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
        # Checkerboard pattern for the floor
        normal = hit_object.normal
        scale = 20.0
        if (int(np.floor(hit_point[0] / scale)) + int(np.floor(hit_point[2] / scale))) % 2 == 0:
            floor_color = np.array([1.0, 1.0, 1.0])
        else:
            floor_color = np.array([0.1, 0.1, 0.1])
        light_dir = normalize(light_pos - hit_point)
        diffuse = max(np.dot(normal, light_dir), 0.0)
        return floor_color * diffuse

# ---------------- Rendering Function ----------------
def render_full():
    """Render a full frame with jittered rays for anti-aliasing, parallelized across CPU cores."""
    ray_origin = CAMERA_POS.copy()
    scene_width = 300
    scene_height = 200
    left = CAMERA_POS[0] - scene_width / 2
    top = CAMERA_POS[1] - scene_height / 2

    # Base pixel centers
    ys, xs = np.indices((RENDER_HEIGHT, RENDER_WIDTH), dtype=np.float32)
    base_u = left + (xs + 0.5) * (scene_width / RENDER_WIDTH)
    base_v = top + (ys + 0.5) * (scene_height / RENDER_HEIGHT)

    # Jitter for anti-aliasing
    jitter_x = np.random.rand(RENDER_HEIGHT, RENDER_WIDTH)
    jitter_y = np.random.rand(RENDER_HEIGHT, RENDER_WIDTH)
    u = base_u + jitter_x * (scene_width / RENDER_WIDTH)
    v = base_v + jitter_y * (scene_height / RENDER_HEIGHT)

    # Sample positions and ray directions
    sample_pos = np.stack((u, v, np.zeros_like(u)), axis=-1)
    ray_dirs = sample_pos - ray_origin
    norms = np.linalg.norm(ray_dirs, axis=-1, keepdims=True)
    ray_dirs = ray_dirs / norms

    # Flatten for parallel processing
    flat_dirs = ray_dirs.reshape(-1, 3)

    # Parallel trace_ray calls using NUM_CPU_CORES
    with concurrent.futures.ProcessPoolExecutor(max_workers=NUM_CPU_CORES) as executor:
        trace_func = functools.partial(trace_ray, ray_origin, depth=MAX_BOUNCES)
        flat_result = list(executor.map(trace_func, flat_dirs, chunksize=100))

    # Reshape and scale to 0-255
    result = np.array(flat_result, dtype=np.float32).reshape(RENDER_HEIGHT, RENDER_WIDTH, 3)
    return result * 255

# ---------------- Main Execution ----------------
if __name__ == '__main__':
    pygame.init()
    screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Continuous Progressive Path Tracer")
    clock = pygame.time.Clock()
    font = pygame.font.SysFont(None, 24)

    # Progressive rendering setup
    accumulated = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.float32)
    current_iter = 0
    start_time = time.time()
    ipm = 0.0  # Initialize IPM

    executor = concurrent.futures.ThreadPoolExecutor(max_workers=1)
    iteration_future = executor.submit(render_full)
    rendering = True

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_r:
                    current_iter = 0
                    accumulated.fill(0)
                    iteration_future = executor.submit(render_full)
                    rendering = True
                    start_time = time.time()
                    ipm = 0.0
                elif event.key == pygame.K_space:
                    rendering = not rendering
                elif event.key == pygame.K_q:
                    running = False

        # Process render iteration
        if rendering and iteration_future.done():
            new_image = iteration_future.result()
            current_iter += 1
            if current_iter == 1:
                accumulated = new_image.copy()
            else:
                accumulated = (accumulated * (current_iter - 1) + new_image) / current_iter
            iteration_future = executor.submit(render_full)

            # Update IPM only when a new frame is loaded
            elapsed_time = time.time() - start_time
            if elapsed_time > 0:
                ipm = current_iter / (elapsed_time / 60)

        # Display the image
        disp_array = np.clip(accumulated, 0, 255).astype(np.uint8)
        lowres_surf = pygame.surfarray.make_surface(disp_array.transpose(1, 0, 2))
        scaled_surf = pygame.transform.scale(lowres_surf, (WINDOW_WIDTH, WINDOW_HEIGHT))
        screen.blit(scaled_surf, (0, 0))

        # UI text (IPM only updates when a new frame is loaded)
        ui_lines = [
            f"Iter/min: {ipm:.2f}",
            f"Total Iterations: {current_iter}",
            "R: Reset  |  Space: Toggle Rendering  |  Q: Quit"
        ]
        for i, line in enumerate(ui_lines):
            text_surf = font.render(line, True, (255, 255, 255))
            screen.blit(text_surf, (10, 10 + i * 20))

        pygame.display.flip()
        clock.tick(60)

    pygame.quit()
    executor.shutdown()