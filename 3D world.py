import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, KEYDOWN, K_SPACE, K_UP, K_DOWN
from OpenGL.GL import *
from OpenGL.GLU import *
import math

def generate_earth_texture():
    """Generate a 256x256 surface with a procedural 'Earth-like' texture.
    The background is deep blue and green ellipses simulate continents."""
    size = 256
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill((0, 0, 128))  # deep blue
    # Draw some green ellipses as continents
    pygame.draw.ellipse(surface, (0, 128, 0), (30, 50, 60, 40))
    pygame.draw.ellipse(surface, (0, 128, 0), (120, 80, 80, 50))
    pygame.draw.ellipse(surface, (0, 128, 0), (70, 150, 100, 60))
    pygame.draw.ellipse(surface, (0, 128, 0), (180, 20, 50, 30))
    return surface

def load_texture_from_surface(surface):
    """Load a PyGame surface as an OpenGL texture."""
    textureData = pygame.image.tostring(surface, "RGBA", True)
    width, height = surface.get_size()
    texID = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texID)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0, GL_RGBA, GL_UNSIGNED_BYTE, textureData)
    return texID

def init_opengl(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(45, width / float(height), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    # Set the camera position (eye at (0,0,5), looking at the origin)
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [0.8, 0.8, 0.8, 1.0])

def draw_earth(texture_id, angle_x, angle_y):
    """Draw the Earth as a textured sphere using the given rotation angles."""
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    # Apply mouse-controlled rotations
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)
    glColor3f(1, 1, 1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    quadric = gluNewQuadric()
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, 1.0, 50, 50)
    gluDeleteQuadric(quadric)

def draw_clouds(angle_x, angle_y):
    """Draw a semi-transparent cloud layer around the Earth."""
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    glDisable(GL_TEXTURE_2D)  # Clouds have no texture
    glColor4f(1, 1, 1, 0.4)
    glLoadIdentity()
    gluLookAt(0, 0, 5, 0, 0, 0, 0, 1, 0)
    # Apply the same base rotations
    glRotatef(angle_y, 1, 0, 0)
    glRotatef(angle_x, 0, 1, 0)
    # Clouds rotate a bit faster than the Earth
    glRotatef(pygame.time.get_ticks() * 0.01, 0, 1, 0)
    quadric = gluNewQuadric()
    gluSphere(quadric, 1.02, 50, 50)
    gluDeleteQuadric(quadric)
    glEnable(GL_TEXTURE_2D)
    glDisable(GL_BLEND)

def get_mouse_ray(mouse_x, mouse_y, screen_width, screen_height):
    """Returns two points (near and far) that define a ray through the clicked point."""
    proj = glGetDoublev(GL_PROJECTION_MATRIX)
    model = glGetDoublev(GL_MODELVIEW_MATRIX)
    viewport = glGetIntegerv(GL_VIEWPORT)
    # Pygame’s y-axis is top-down; OpenGL expects bottom-up.
    win_y = screen_height - mouse_y
    near = gluUnProject(mouse_x, win_y, 0.0, model, proj, viewport)
    far = gluUnProject(mouse_x, win_y, 1.0, model, proj, viewport)
    return near, far

def vector_normalize(v):
    length = math.sqrt(v[0]**2 + v[1]**2 + v[2]**2)
    if length == 0:
        return (0, 0, 0)
    return (v[0]/length, v[1]/length, v[2]/length)

def intersect_ray_sphere(ray_origin, ray_direction, sphere_center, sphere_radius):
    """Compute the intersection point (if any) of a ray and a sphere."""
    ox, oy, oz = ray_origin
    dx, dy, dz = ray_direction
    cx, cy, cz = sphere_center
    ocx, ocy, ocz = ox - cx, oy - cy, oz - cz
    a = dx*dx + dy*dy + dz*dz
    b = 2 * (ocx*dx + ocy*dy + ocz*dz)
    c = ocx*ocx + ocy*ocy + ocz*ocz - sphere_radius * sphere_radius
    discriminant = b*b - 4*a*c
    if discriminant < 0:
        return None
    t = (-b - math.sqrt(discriminant)) / (2 * a)
    if t < 0:
        t = (-b + math.sqrt(discriminant)) / (2 * a)
    if t < 0:
        return None
    intersection = (ox + t*dx, oy + t*dy, oz + t*dz)
    return intersection

def compute_lat_long(point):
    """Convert a point on the unit sphere into latitude and longitude (degrees)."""
    x, y, z = point
    lat = math.degrees(math.asin(y))
    lon = math.degrees(math.atan2(z, x))
    return lat, lon

def main():
    pygame.init()
    screen_width, screen_height = 800, 600
    screen = pygame.display.set_mode((screen_width, screen_height), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Interactive 3D Earth Simulator")
    init_opengl(screen_width, screen_height)

    # Create the procedural Earth texture and load it.
    earth_surface = generate_earth_texture()
    texture_id = load_texture_from_surface(earth_surface)

    clock = pygame.time.Clock()
    angle_x = 0    # rotation around y-axis (horizontal)
    angle_y = 0    # rotation around x-axis (vertical)
    auto_rotate = True
    rotation_speed = 0.3  # degrees per frame
    paused = False
    dragging = False
    last_mouse_pos = (0, 0)
    sim_time = 0.0  # simulation time for sun orbit

    running = True
    while running:
        dt = clock.tick(60) / 1000.0  # Delta time in seconds
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == MOUSEBUTTONDOWN:
                if event.button == 1:  # Left click: start dragging for manual rotation
                    dragging = True
                    last_mouse_pos = pygame.mouse.get_pos()
                elif event.button == 3:  # Right click: pick a point on Earth
                    mouse_pos = pygame.mouse.get_pos()
                    near, far = get_mouse_ray(mouse_pos[0], mouse_pos[1], screen_width, screen_height)
                    ray_dir = (far[0] - near[0], far[1] - near[1], far[2] - near[2])
                    ray_dir = vector_normalize(ray_dir)
                    intersection = intersect_ray_sphere(near, ray_dir, (0, 0, 0), 1)
                    if intersection:
                        lat, lon = compute_lat_long(intersection)
                        print(f"Clicked on Earth: lat = {lat:.2f}°, lon = {lon:.2f}°")
            elif event.type == MOUSEBUTTONUP:
                if event.button == 1:
                    dragging = False
            elif event.type == MOUSEMOTION:
                if dragging:
                    x, y = pygame.mouse.get_pos()
                    dx = x - last_mouse_pos[0]
                    dy = y - last_mouse_pos[1]
                    angle_x += dx * 0.5
                    angle_y += dy * 0.5
                    last_mouse_pos = (x, y)
            elif event.type == KEYDOWN:
                if event.key == K_SPACE:
                    paused = not paused
                elif event.key == K_UP:
                    rotation_speed += 0.1
                elif event.key == K_DOWN:
                    rotation_speed = max(0.1, rotation_speed - 0.1)
        
        if not paused:
            sim_time += dt
            if auto_rotate:
                angle_x += rotation_speed

        # Update the sun's position for dynamic lighting.
        # The sun moves in the XZ plane around the Earth.
        sun_angle = sim_time * 20  # Adjust orbit speed factor as needed.
        sun_x = 5 * math.cos(math.radians(sun_angle))
        sun_z = 5 * math.sin(math.radians(sun_angle))
        glLightfv(GL_LIGHT0, GL_POSITION, [sun_x, 0, sun_z, 1.0])

        # Draw the Earth and then the cloud layer.
        draw_earth(texture_id, angle_x, angle_y)
        draw_clouds(angle_x, angle_y)
        pygame.display.flip()

    pygame.quit()

if __name__ == '__main__':
    main()
