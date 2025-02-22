import numpy as np
import pygame
import time
from concurrent.futures import ThreadPoolExecutor

# ---------------- Parameters ----------------
MAX_BOUNCES = 3  # Maximum recursion depth (number of light bounces)

# ---------------- Resolution & Camera Settings ----------------
WINDOW_HEIGHT = 500   # Display window height
WINDOW_WIDTH = int(WINDOW_HEIGHT * 1.5)   # Display window width
RENDER_HEIGHT = 500   # Low-res render height
RENDER_WIDTH = int(RENDER_HEIGHT * 1.5)    # Low-res render width
CAMERA_POS = np.array([150, 100, -1000], dtype=np.float32)  # Explicit camera position in low-res space

# ---------------- Utility Functions ----------------
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

# ---------------- Pygame & Render Setup ----------------
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Continuous Progressive Path Tracer")
clock = pygame.time.Clock()
font = pygame.font.SysFont(None, 24)

# ---------------- Scene Setup ----------------
# Define scene objects in a low-res coordinate system (300x200).
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
    # Extra objects:
    Sphere([150, 40, 80], 30, Material([1.0, 1.0, 1.0], is_glass=True, ior=1.5, reflectivity=0.1)),
    Sphere([220, 40, 120], 30, Material([0.9, 0.9, 0.9], reflectivity=0.3)),
    Sphere([75, 30, 170], 30, Material([0.5, 0.5, 0.5], reflectivity=0.1)),
    Sphere([225, 30, 170], 30, Material([0.7, 0.3, 0.3], reflectivity=0.1)),
]

planes = [
    # Floor plane with a checkerboard pattern.
    Plane(point=[0, 160, 0], normal=[0, -1, 0],
          material=Material([0.8, 0.8, 0.8], reflectivity=0.0))
]

# Single light source in low-res coordinates.
light_pos = np.array([300 * 0.5, 200 * 0.2, -150], dtype=np.float32)

# ---------------- Ray Tracing Function ----------------
def trace_ray(ray_origin, ray_dir, depth=MAX_BOUNCES):
    if depth <= 0:
        return np.zeros(3)
    
    closest_t = np.inf
    hit_object = None
    hit_type = None  # 'sphere' or 'plane'
    
    # Check sphere intersections
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
                
    # Check plane intersections
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
        # For the floor (plane), use a checkerboard pattern.
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
    result = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.float32)
    ray_origin = CAMERA_POS.copy()
    
    # Define the world dimensions for the image plane.
    # These remain fixed so the scene stays the same regardless of render resolution.
    scene_width = 300
    scene_height = 200
    
    # Compute the left and top of the image plane in world coordinates,
    # centering it at (CAMERA_POS[0], CAMERA_POS[1]).
    left = CAMERA_POS[0] - scene_width / 2
    top = CAMERA_POS[1] - scene_height / 2
    
    for y in range(RENDER_HEIGHT):
        for x in range(RENDER_WIDTH):
            # Random jitter for anti-aliasing
            jitter_x, jitter_y = np.random.rand(), np.random.rand()
            
            # Map the pixel coordinate to a world coordinate on the image plane
            u = left + (x + jitter_x) * (scene_width / RENDER_WIDTH)
            v = top + (y + jitter_y) * (scene_height / RENDER_HEIGHT)
            sample_pos = np.array([u, v, 0], dtype=np.float32)
            
            # Calculate the ray direction so that the camera always points towards the center of the scene.
            ray_dir = normalize(sample_pos - ray_origin)
            color = trace_ray(ray_origin, ray_dir, MAX_BOUNCES)
            result[y, x] = color * 255
    return result


# ---------------- Progressive Rendering Setup ----------------
accumulated = np.zeros((RENDER_HEIGHT, RENDER_WIDTH, 3), dtype=np.float32)
current_iter = 0

executor = ThreadPoolExecutor(max_workers=1)
iteration_future = executor.submit(render_full)
rendering = True
last_update = time.time()

# ---------------- Main Loop ----------------
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
            elif event.key == pygame.K_SPACE:
                rendering = not rendering
            elif event.key == pygame.K_q:
                running = False

    if rendering and iteration_future.done():
        new_image = iteration_future.result()
        current_iter += 1
        if current_iter == 1:
            accumulated = new_image.copy()
        else:
            accumulated = (accumulated * (current_iter - 1) + new_image) / current_iter
        iteration_future = executor.submit(render_full)
    
    disp_array = np.clip(accumulated, 0, 255).astype(np.uint8)
    lowres_surf = pygame.surfarray.make_surface(disp_array.transpose(1, 0, 2))
    scaled_surf = pygame.transform.scale(lowres_surf, (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(scaled_surf, (0, 0))
    
    current_time = time.time()
    fps = 1.0 / (current_time - last_update + 1e-6)
    last_update = current_time
    ui_lines = [
        f"FPS: {fps:.1f}",
        f"Iterations: {current_iter}",
        "R: Reset  |  Space: Toggle Rendering  |  Q: Quit"
    ]
    for i, line in enumerate(ui_lines):
        text_surf = font.render(line, True, (255, 255, 255))
        screen.blit(text_surf, (10, 10 + i * 20))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()
executor.shutdown()
