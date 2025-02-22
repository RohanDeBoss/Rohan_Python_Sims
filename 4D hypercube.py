import pygame
import numpy as np
from math import sin, cos, pi
from pygame.locals import *

# Initialize Pygame
pygame.init()

# Set up the display
WIDTH = 800
HEIGHT = 600
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Interactive 4D Hypercube (Tesseract)")

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)
BLUE = (0, 0, 255)
GREEN = (0, 255, 0)

# Hypercube vertices (4D coordinates)
# A tesseract has 16 vertices
vertices_4d = np.array([
    [x, y, z, w]
    for x in [-1, 1]
    for y in [-1, 1]
    for z in [-1, 1]
    for w in [-1, 1]
])

# Edges connecting the vertices
edges = []
for i in range(len(vertices_4d)):
    for j in range(i + 1, len(vertices_4d)):
        # Connect vertices that differ in exactly one coordinate
        if sum(abs(vertices_4d[i][k] - vertices_4d[j][k]) for k in range(4)) == 2:
            edges.append((i, j))

# Rotation angles for different planes
angles = {
    'xy': 0,
    'xz': 0,
    'xw': 0,
    'yz': 0,
    'yw': 0,
    'zw': 0
}

def rotation_matrix_4d(angle, plane):
    """Create a 4D rotation matrix for the given plane"""
    c, s = cos(angle), sin(angle)
    matrix = np.eye(4)
    
    if plane == 'xy':
        matrix[0:2, 0:2] = [[c, -s], [s, c]]
    elif plane == 'xz':
        matrix[0, 0], matrix[0, 2] = c, -s
        matrix[2, 0], matrix[2, 2] = s, c
    elif plane == 'xw':
        matrix[0, 0], matrix[0, 3] = c, -s
        matrix[3, 0], matrix[3, 3] = s, c
    elif plane == 'yz':
        matrix[1, 1], matrix[1, 2] = c, -s
        matrix[2, 1], matrix[2, 2] = s, c
    elif plane == 'yw':
        matrix[1, 1], matrix[1, 3] = c, -s
        matrix[3, 1], matrix[3, 3] = s, c
    elif plane == 'zw':
        matrix[2, 2], matrix[2, 3] = c, -s
        matrix[3, 2], matrix[3, 3] = s, c
    
    return matrix

def project_4d_to_3d(points_4d, distance=4):
    """Project 4D points to 3D using perspective projection"""
    points_3d = []
    for point in points_4d:
        w = distance / (distance - point[3])  # Perspective projection
        x = point[0] * w
        y = point[1] * w
        z = point[2] * w
        points_3d.append([x, y, z])
    return np.array(points_3d)

def project_3d_to_2d(points_3d, distance=5):
    """Project 3D points to 2D using perspective projection"""
    points_2d = []
    for point in points_3d:
        f = distance / (distance - point[2])  # Perspective projection
        x = point[0] * f * 100 + WIDTH/2
        y = point[1] * f * 100 + HEIGHT/2
        points_2d.append([x, y])
    return np.array(points_2d)

# Main loop
clock = pygame.time.Clock()
running = True

# Rotation controls
rotation_speed = 0.02
active_rotations = set()

while running:
    for event in pygame.event.get():
        if event.type == QUIT:
            running = False
        elif event.type == KEYDOWN:
            # Add rotation plane when key is pressed
            if event.key == K_q:  # XY rotation
                active_rotations.add('xy')
            elif event.key == K_w:  # XZ rotation
                active_rotations.add('xz')
            elif event.key == K_e:  # XW rotation
                active_rotations.add('xw')
            elif event.key == K_a:  # YZ rotation
                active_rotations.add('yz')
            elif event.key == K_s:  # YW rotation
                active_rotations.add('yw')
            elif event.key == K_d:  # ZW rotation
                active_rotations.add('zw')
        elif event.type == KEYUP:
            # Remove rotation plane when key is released
            if event.key == K_q:
                active_rotations.discard('xy')
            elif event.key == K_w:
                active_rotations.discard('xz')
            elif event.key == K_e:
                active_rotations.discard('xw')
            elif event.key == K_a:
                active_rotations.discard('yz')
            elif event.key == K_s:
                active_rotations.discard('yw')
            elif event.key == K_d:
                active_rotations.discard('zw')

    # Update rotations
    for plane in active_rotations:
        angles[plane] += rotation_speed

    # Apply all rotations
    rotated_points = vertices_4d.copy()
    for plane, angle in angles.items():
        rotation = rotation_matrix_4d(angle, plane)
        rotated_points = np.dot(rotated_points, rotation)

    # Project 4D -> 3D -> 2D
    points_3d = project_4d_to_3d(rotated_points)
    points_2d = project_3d_to_2d(points_3d)

    # Clear screen
    screen.fill(BLACK)

    # Draw edges
    for edge in edges:
        start = points_2d[edge[0]].astype(int)
        end = points_2d[edge[1]].astype(int)
        pygame.draw.line(screen, WHITE, start, end, 2)

    # Draw vertices
    for point in points_2d:
        pygame.draw.circle(screen, RED, point.astype(int), 5)

    # Draw instructions
    font = pygame.font.Font(None, 24)
    instructions = [
        "Controls:",
        "Q: Rotate XY plane",
        "W: Rotate XZ plane",
        "E: Rotate XW plane",
        "A: Rotate YZ plane",
        "S: Rotate YW plane",
        "D: Rotate ZW plane"
    ]
    for i, text in enumerate(instructions):
        text_surface = font.render(text, True, GREEN)
        screen.blit(text_surface, (10, 10 + i * 20))

    pygame.display.flip()
    clock.tick(60)

pygame.quit()