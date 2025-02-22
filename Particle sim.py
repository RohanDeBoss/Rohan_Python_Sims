import pygame
import random
import math
from collections import defaultdict

# Initialize pygame
pygame.init()

# Constants
WIDTH, HEIGHT = 800, 600
PARTICLE_RADIUS = 5
GRAVITY = 0.11
DENSITY_RADIUS = 19
MAX_PARTICLES = 2000
ADD_RATE = 9
GRID_SIZE = DENSITY_RADIUS * 1.5
ELASTICITY = 0.7
AIR_RESISTANCE = 0.99

# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
RED = (255, 0, 0)

# Screen setup
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Optimized Particle Simulation")
font = pygame.font.Font(None, 30)

# Particle class
class Particle:
    __slots__ = ('x', 'y', 'vx', 'vy', 'density')
    
    def __init__(self, x, y):
        self.x = x
        self.y = y
        self.vx = random.uniform(-1, 1)
        self.vy = random.uniform(-1, 1)
        self.density = 1

    def update(self, grid, particles):
        self.vx *= AIR_RESISTANCE
        self.vy *= AIR_RESISTANCE
        self.vy += GRAVITY
        self.x += self.vx
        self.y += self.vy
        
        if self.x - PARTICLE_RADIUS < 0 or self.x + PARTICLE_RADIUS > WIDTH:
            self.vx *= -ELASTICITY
        if self.y - PARTICLE_RADIUS < 0 or self.y + PARTICLE_RADIUS > HEIGHT:
            self.vy *= -ELASTICITY
        
        self.x = max(PARTICLE_RADIUS, min(WIDTH - PARTICLE_RADIUS, self.x))
        self.y = max(PARTICLE_RADIUS, min(HEIGHT - PARTICLE_RADIUS, self.y))
        
        grid_x, grid_y = int(self.x // GRID_SIZE), int(self.y // GRID_SIZE)
        self.density = 0
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                for p in grid.get((grid_x + dx, grid_y + dy), []):
                    if p is not self:
                        dist_sq = (p.x - self.x) ** 2 + (p.y - self.y) ** 2
                        if dist_sq < DENSITY_RADIUS ** 2:
                            self.density += math.exp(-dist_sq / (DENSITY_RADIUS ** 2))
        
        for p in grid.get((grid_x, grid_y), []):
            if p is not self and math.hypot(p.x - self.x, p.y - self.y) < PARTICLE_RADIUS * 2:
                dx, dy = p.x - self.x, p.y - self.y
                dist = math.hypot(dx, dy) or 1
                overlap = PARTICLE_RADIUS * 2 - dist
                self.x -= (dx / dist) * overlap / 2
                self.y -= (dy / dist) * overlap / 2
                p.x += (dx / dist) * overlap / 2
                p.y += (dy / dist) * overlap / 2
                self.vx, p.vx = p.vx, self.vx
                self.vy, p.vy = p.vy, self.vy

    def draw(self, screen):
        color = (min(255, int(self.density * 50)), 0, 255 - min(255, int(self.density * 50)))
        pygame.draw.circle(screen, color, (int(self.x), int(self.y)), PARTICLE_RADIUS)

# Particle list
particles = []
adding_particles = False
clock = pygame.time.Clock()

# Main loop
running = True
while running:
    screen.fill(WHITE)
    mouse_x, mouse_y = pygame.mouse.get_pos()
    
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False
        elif event.type == pygame.MOUSEBUTTONDOWN:
            if event.button == 1:
                adding_particles = True
            elif event.button == 3:
                particles = [p for p in particles if math.hypot(p.x - event.pos[0], p.y - event.pos[1]) > 10]
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                adding_particles = False
        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_r:
                particles.clear()
    
    if adding_particles and len(particles) < MAX_PARTICLES:
        particles.extend(Particle(mouse_x, mouse_y) for _ in range(ADD_RATE))
    
    grid = defaultdict(list)
    for p in particles:
        grid_x, grid_y = int(p.x // GRID_SIZE), int(p.y // GRID_SIZE)
        grid[(grid_x, grid_y)].append(p)
    
    for p in particles:
        p.update(grid, particles)
        p.draw(screen)
    
    fps = int(clock.get_fps())
    text_particles = font.render(f'Particles: {len(particles)} / {MAX_PARTICLES}', True, BLACK)
    text_fps = font.render(f'FPS: {fps}', True, RED)
    screen.blit(text_particles, (10, 10))
    screen.blit(text_fps, (10, 40))
    
    pygame.display.flip()
    clock.tick(60)

pygame.quit()