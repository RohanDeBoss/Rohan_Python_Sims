import pygame
import random
import sys
from typing import Tuple, Optional, List, Dict, Set
from array import array

# Global Constants
SIM_WIDTH, UI_WIDTH, SCREEN_HEIGHT = 800, 270, 600
SCREEN_WIDTH = SIM_WIDTH + UI_WIDTH
CELL_SIZE = 4
GRID_WIDTH, GRID_HEIGHT = SIM_WIDTH // CELL_SIZE, SCREEN_HEIGHT // CELL_SIZE
FPS = 60

# Colors
BLACK, WHITE = array('B', [0, 0, 0]), array('B', [255, 255, 255])
GRAY, DARKGRAY = array('B', [100, 100, 100]), array('B', [50, 50, 50])
RED, GREEN, BLUE = array('B', [255, 0, 0]), array('B', [0, 255, 0]), array('B', [0, 0, 255])
UI_BG, TITLE_COLOR = array('B', [30, 30, 30]), array('B', [200, 200, 240])
SELECTED_COLOR = array('B', [255, 215, 0])

# Pre-defined direction lists
DIRECTIONS = [-1, 1]
NEIGHBORS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

class ParticleType:
    __slots__ = ('name', 'base_color', 'density', 'is_liquid', 'gravity', 'friction',
                 'viscosity', 'flammable', 'base_temp', 'corrosive', 'freezing_point',
                 'melting_point', 'thermal_conductivity')

    def __init__(self, name: str, color: Tuple[int, int, int], density: float, 
                 is_liquid: bool, gravity: float, friction: float, 
                 viscosity: float = 0, flammable: bool = False, 
                 base_temp: float = 20.0, corrosive: bool = False,
                 freezing_point: float = -float('inf'), melting_point: float = float('inf'),
                 thermal_conductivity: float = 0.2):
        self.name = name
        self.base_color = array('B', color)
        self.density = density
        self.is_liquid = is_liquid
        self.gravity = gravity
        self.friction = friction
        self.viscosity = viscosity
        self.flammable = flammable
        self.base_temp = base_temp
        self.corrosive = corrosive
        self.freezing_point = freezing_point
        self.melting_point = melting_point
        self.thermal_conductivity = thermal_conductivity

# Global particle types (not in UI hotbar)
STEAM = ParticleType("Steam", (200, 200, 200), 0.1, True, -1, 0.02, 0.1, False, 100.0, thermal_conductivity=0.1)
SMOKE = ParticleType("Smoke", (70, 70, 70), 0.2, True, -1, 0.03, 0.1, True, 80.0, thermal_conductivity=0.05)
ASH = ParticleType("Ash", (120, 120, 120), 0.5, False, 1, 0.1, 0, False, 40.0, thermal_conductivity=0.15)
MERCURY_VAPOR = ParticleType("Mercury Vapor", (150, 150, 180), 0.15, True, -1, 0.02, 0.1, False, 357.0, thermal_conductivity=0.1)
SOLID_MERCURY = ParticleType("Solid Mercury", (160, 160, 180), 13.5, False, 1, 0.1, 0, False, -38.8, melting_point=357, thermal_conductivity=0.8)
DILUTED_ACID = ParticleType("Diluted Acid", (100, 255, 100), 1.1, True, 1, 0.05, 0.15, False, 20.0, corrosive=True, thermal_conductivity=0.3)

# Main particle types list (placeable ones)
PARTICLE_TYPES = [
    SAND    := ParticleType("Sand",    (220, 180, 100), 2.0, False, 1, 0.1, melting_point=1600, thermal_conductivity=0.25),
    WATER   := ParticleType("Water",   (0, 100, 200),   1.0, True, 1, 0.05, 0.15, freezing_point=0, melting_point=100, thermal_conductivity=0.6),
    OIL     := ParticleType("Oil",     (90, 60, 30),    0.8, True, 1, 0.03, 0.15, True, 20, melting_point=200, thermal_conductivity=0.2),
    FIRE    := ParticleType("Fire",    (255, 140, 40),  0.2, False, -1, 0.1, 0, False, 400.0, thermal_conductivity=0.1),
    ACID    := ParticleType("Acid",    (50, 255, 50),   1.2, True, 1, 0.05, 0.2, False, 20.0, True, thermal_conductivity=0.4),
    LAVA    := ParticleType("Lava",    (255, 60, 60),   3.0, True, 1, 0.2, 0.35, False, 1000.0, thermal_conductivity=0.3),
    ICE     := ParticleType("Ice",     (180, 240, 255), 1.5, False, 1, 0.3, 0, False, -10.0, melting_point=0, thermal_conductivity=0.5),
    PLASMA  := ParticleType("Plasma",  (180, 0, 255),   0.05, False, -1, 0.02, 0, False, 5000.0, thermal_conductivity=0.8),
    MERCURY := ParticleType("Mercury", (180, 180, 200), 13.5, True, 1, 0.01, 0.3, False, 20.0, freezing_point=-38.8, melting_point=357, thermal_conductivity=0.8),
    CRYSTAL := ParticleType("Crystal", (80, 220, 200),  5.0, False, 0, 0.0, base_temp=20.0, thermal_conductivity=0.4),
    VOID    := ParticleType("Void",    (80, 60, 120),   5.0, False, 0, 0.0, base_temp=0.0, thermal_conductivity=1.0),
    GLASS   := ParticleType("Glass",   (220, 220, 240), 2.5, False, 0, 0.0, base_temp=20.0, melting_point=1600, thermal_conductivity=0.3),
]

INTERACTION_CACHE: Dict[Tuple[ParticleType, ParticleType], Optional[Tuple[Optional[ParticleType], Optional[ParticleType]]]] = {
    # FIRE Interactions
    (FIRE, WATER): (None, STEAM),           # Fire extinguished, water to steam
    (FIRE, OIL): (FIRE, SMOKE),             # Fire spreads, oil to smoke
    (FIRE, ICE): (None, WATER),             # Fire melts ice, extinguishes
    (FIRE, ACID): (SMOKE, DILUTED_ACID),    # Fire produces smoke, acid dilutes
    (FIRE, PLASMA): (PLASMA, PLASMA),       # Fire merges into plasma
    (FIRE, VOID): (VOID, None),             # Void consumes fire
    (FIRE, MERCURY_VAPOR): (FIRE, SMOKE),   # Mercury vapor burns

    # ACID Interactions
    (ACID, WATER): (DILUTED_ACID, None),    # Acid dilutes in water
    (ACID, SAND): (None, SMOKE),            # Acid reacts with sand, produces smoke
    (ACID, ICE): (None, WATER),             # Acid melts ice to water
    (ACID, MERCURY): (None, MERCURY_VAPOR), # Acid produces mercury vapor
    (ACID, CRYSTAL): (None, SMOKE),         # Acid dissolves crystal, releases smoke
    (ACID, GLASS): (None, SAND),            # Acid etches glass to sand
    (ACID, SOLID_MERCURY): (None, MERCURY_VAPOR), # Acid vaporizes solid mercury

    # LAVA Interactions
    (LAVA, WATER): (GLASS, STEAM),          # Lava cools to glass, water to steam
    (LAVA, ICE): (GLASS, WATER),            # Lava cools to glass, ice to water
    (LAVA, OIL): (FIRE, SMOKE),             # Lava ignites oil to fire and smoke
    (LAVA, SAND): (None, SAND),             # Lava heats sand but doesn’t transform it
    (LAVA, CRYSTAL): (GLASS, None),         # Lava melts crystal to glass
    (LAVA, SOLID_MERCURY): (MERCURY, None), # Lava melts solid mercury

    # WATER Interactions
    (WATER, FIRE): (STEAM, None),           # Water to steam, fire extinguished
    (WATER, MERCURY_VAPOR): (MERCURY, None), # Mercury vapor condenses to liquid

    # OIL Interactions
    (OIL, FIRE): (SMOKE, FIRE),             # Oil burns, fire persists

    # ICE Interactions
    (ICE, LAVA): (WATER, GLASS),            # Ice melts to water, lava to glass

    # PLASMA Interactions
    (PLASMA, WATER): (STEAM, STEAM),        # Plasma vaporizes water
    (PLASMA, ICE): (STEAM, WATER),          # Plasma melts ice, some to steam
    (PLASMA, SAND): (LAVA, SMOKE),          # Plasma melts sand to lava with smoke
    (PLASMA, CRYSTAL): (PLASMA, SMOKE),     # Plasma ionizes crystal
    (PLASMA, GLASS): (LAVA, None),          # Plasma remelts glass to lava
    (PLASMA, VOID): (VOID, FIRE),           # Void downgrades plasma to fire

    # VOID Interactions
    (VOID, FIRE): (VOID, None),             # Void consumes fire
    (VOID, ACID): (VOID, WATER),            # Void neutralizes acid to water
    (VOID, PLASMA): (VOID, FIRE),           # Void reduces plasma to fire

    # DILUTED_ACID Interactions
    (DILUTED_ACID, CRYSTAL): (None, SAND),  # Diluted acid degrades crystal to sand
    (DILUTED_ACID, GLASS): (None, SAND),    # Diluted acid etches glass to sand
}

class Particle:
    __slots__ = ('type', 'x', 'y', 'updated', 'stationary', 'temperature', 
                 'lifetime', 'base_color', 'color')

    def __init__(self, p_type: ParticleType, x: int, y: int):
        self.type = p_type
        self.x, self.y = x, y
        self.updated = False
        self.stationary = p_type.gravity == 0
        self.temperature = p_type.base_temp
        self.lifetime = 30 if p_type in (FIRE, PLASMA) else None
        self.base_color = p_type.base_color
        self.color = self._vary_color()

    def _vary_color(self) -> array:
        if self.type.name == "Fire":
            r = min(255, max(200, self.base_color[0] + random.randint(-20, 20)))
            g = min(180, max(100, self.base_color[1] + random.randint(-20, 20)))
            b = min(80, max(20, self.base_color[2] + random.randint(-20, 20)))
            return array('B', [r, g, b])
        variation = min(15, max(5, int(self.base_color[0] * 0.05)))
        return array('B', [max(0, min(255, c + random.randint(-variation, variation))) 
                         for c in self.base_color])

    def reset(self):
        self.updated = False

class Simulation:
    def __init__(self, sim_width: int, height: int, cell_size: int):
        self.sim_width, self.height, self.cell_size = sim_width, height, cell_size
        self.grid_width, self.grid_height = sim_width // cell_size, height // cell_size
        self.grid = [[None for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        
        self.paused = False
        self.current_particle = PARTICLE_TYPES[0]
        self.brush_size = 3
        self.simulation_speed = 1
        self.show_grid = False
        self.show_temp = False
        self.mouse_held = False
        self.hotbar_slots = [(p, pygame.Rect(0, 0, 30, 30)) for p in PARTICLE_TYPES]
        
        self.font = pygame.font.SysFont("Arial", 16)
        self.ui_font = pygame.font.SysFont("Arial", 12)
        self.title_font = pygame.font.SysFont("Arial", 20, bold=True)
        self.background = pygame.Surface((self.sim_width, self.height))
        self.background.fill(tuple(BLACK))
        self.grid_surface = pygame.Surface((self.sim_width, self.height))
        self.grid_surface.fill(tuple(BLACK))
        self.active_set: Set[Tuple[int, int]] = set()
        self.active_types: Set[ParticleType] = set()
        self.all_particles: Set[Tuple[int, int]] = set()

    def in_bounds(self, x: int, y: int) -> bool:
        return 0 <= x < self.grid_width and 0 <= y < self.grid_height

    def mark_active(self, x: int, y: int, active_set: Set[Tuple[int, int]] = None):
        if active_set is None:
            active_set = self.active_set
        if self.in_bounds(x, y):
            active_set.add((x, y))
            if self.grid[x][y]:
                self.active_types.add(self.grid[x][y].type)
                self.all_particles.add((x, y))
            for dx, dy in NEIGHBORS:
                nx, ny = x + dx, y + dy
                if self.in_bounds(nx, ny):
                    active_set.add((nx, ny))
                    if self.grid[nx][ny]:
                        self.active_types.add(self.grid[nx][ny].type)
                        self.all_particles.add((nx, ny))

    def add_particle(self, grid_x: int, grid_y: int, p_type: Optional[ParticleType] = None):
        if not self.in_bounds(grid_x, grid_y) or self.grid[grid_x][grid_y]:
            return
        p_type = p_type or self.current_particle
        self.grid[grid_x][grid_y] = Particle(p_type, grid_x, grid_y)
        self.mark_active(grid_x, grid_y)

    def remove_particle(self, grid_x: int, grid_y: int):
        if self.in_bounds(grid_x, grid_y) and self.grid[grid_x][grid_y]:
            self.grid[grid_x][grid_y] = None
            self.mark_active(grid_x, grid_y)
            self.all_particles.discard((grid_x, grid_y))

    def clear_grid(self):
        self.grid = [[None for _ in range(self.grid_height)] for _ in range(self.grid_width)]
        self.active_set.clear()
        self.active_types.clear()
        self.all_particles.clear()
        self.grid_surface.fill(tuple(BLACK))

    def reset_boundaries(self):
        self.clear_grid()
        for x in range(self.grid_width):
            self.add_particle(x, self.grid_height - 1, GLASS)
        for y in range(self.grid_height):
            self.add_particle(0, y, GLASS)
            self.add_particle(self.grid_width - 1, y, GLASS)

    def move_particle(self, old_x: int, old_y: int, new_x: int, new_y: int, active_set: Set[Tuple[int, int]]):
        particle = self.grid[old_x][old_y]
        if not particle or particle.type.gravity == 0:
            return
        self.grid[new_x][new_y] = particle
        self.grid[old_x][old_y] = None
        particle.x, particle.y = new_x, new_y
        particle.updated = True
        particle.stationary = False
        self.mark_active(new_x, new_y, active_set)
        self.mark_active(old_x, old_y, active_set)
        self.all_particles.add((new_x, new_y))
        self.all_particles.discard((old_x, old_y))

    def swap_particles(self, x1: int, y1: int, x2: int, y2: int, active_set: Set[Tuple[int, int]]):
        p1, p2 = self.grid[x1][y1], self.grid[x2][y2]
        if (p1 and p1.type.gravity == 0) or (p2 and p2.type.gravity == 0):
            return
        self.grid[x1][y1], self.grid[x2][y2] = p2, p1
        if p1:
            p1.x, p1.y = x2, y2
            p1.updated = True
            p1.stationary = False
        if p2:
            p2.x, p2.y = x1, y1
            p2.updated = True
            p2.stationary = False
        self.mark_active(x1, y1, active_set)
        self.mark_active(x2, y2, active_set)

    def can_fall(self, x: int, y: int, direction: int) -> bool:
        target_y = y + direction
        if not self.in_bounds(x, target_y):
            return False
        particle = self.grid[x][y]
        if not particle or particle.type.gravity == 0:
            return False
        target = self.grid[x][target_y]
        return target is None or (target.type.is_liquid and particle.type.density > target.type.density)

    def update_particle(self, x: int, y: int, active_set: Set[Tuple[int, int]]):
        particle = self.grid[x][y]
        if not particle or particle.updated or particle.type.gravity == 0:
            return

        p_type = particle.type
        density = p_type.density
        direction = 1 if p_type.gravity >= 0 else -1
        moved = False

        gw, gh = self.grid_width, self.grid_height

        if particle.stationary:
            if not self.can_fall(x, y, direction):
                if not any(0 <= x+dx < gw and 0 <= y+dy < gh and 
                          self.grid[x+dx][y+dy] and not self.grid[x+dx][y+dy].stationary 
                          for dx, dy in NEIGHBORS):
                    return
            else:
                particle.stationary = False

        if p_type.is_liquid:
            new_y = y + direction
            flow_dirs = [-1, 1] if random.random() < 0.5 else [1, -1]
            viscosity = p_type.viscosity
            
            if 0 <= new_y < gh and self.can_fall(x, y, direction):
                self.move_particle(x, y, x, new_y, active_set)
                return

            flow_strength = 1.0 - viscosity
            for dx in flow_dirs:
                nx = x + dx
                if not (0 <= nx < gw):
                    continue
                    
                if 0 <= new_y < gh:
                    target = self.grid[nx][new_y]
                    if target is None:
                        if random.random() < flow_strength:
                            self.move_particle(x, y, nx, new_y, active_set)
                            return
                    elif target.type.is_liquid and density > target.type.density:
                        if random.random() < flow_strength * 0.8:
                            self.swap_particles(x, y, nx, new_y, active_set)
                            return

                target = self.grid[nx][y]
                if target is None:
                    can_flow_down = (0 <= new_y < gh and 
                                   (self.grid[nx][new_y] is None or 
                                    (self.grid[nx][new_y].type.is_liquid and 
                                     density > self.grid[nx][new_y].type.density)))
                    flow_chance = flow_strength * (1.5 if can_flow_down else 0.5)
                    if random.random() < flow_chance:
                        self.move_particle(x, y, nx, y, active_set)
                        return
                elif target.type.is_liquid and density >= target.type.density:
                    if random.random() < flow_strength * 0.3:
                        self.swap_particles(x, y, nx, y, active_set)
                        return
        else:
            new_y = y + direction
            if self.can_fall(x, y, direction):
                self.move_particle(x, y, x, new_y, active_set)
                moved = True
            else:
                flow_dirs = [-1, 1] if random.random() < 0.5 else [1, -1]
                for dx in flow_dirs:
                    nx = x + dx
                    if (0 <= nx < gw and 0 <= new_y < gh and 
                        not self.grid[nx][new_y]):
                        self.move_particle(x, y, nx, new_y, active_set)
                        moved = True
                        break

        self.handle_physics(x, y, active_set)
        particle.updated = True
        if not moved and not particle.lifetime and p_type not in (FIRE, PLASMA, ACID, DILUTED_ACID):
            if not self.can_fall(x, y, direction):
                particle.stationary = True

    def handle_physics(self, x: int, y: int, active_set: Set[Tuple[int, int]]):
        particle = self.grid[x][y]
        if not particle:
            return

        changed = False
        
        # Temperature-based phase changes
        if particle.type == WATER:
            if particle.temperature <= 0:
                self.grid[x][y] = Particle(ICE, x, y)
                changed = True
            elif particle.temperature >= 100:
                self.grid[x][y] = Particle(STEAM, x, y)
                changed = True
        elif particle.type == STEAM:
            if particle.temperature < 100:
                condense_chance = (100 - particle.temperature) / 100
                if random.random() < condense_chance:
                    self.grid[x][y] = Particle(WATER, x, y)
                    changed = True
        elif particle.type == ICE:
            if particle.temperature >= 0:
                self.grid[x][y] = Particle(WATER, x, y)
                changed = True
        elif particle.type == OIL and particle.temperature >= 200:
            if random.random() < 0.5:
                self.grid[x][y] = Particle(FIRE, x, y)
            else:
                self.grid[x][y] = Particle(SMOKE, x, y)
            changed = True
        elif particle.type == MERCURY:
            if particle.temperature <= -38.8:
                self.grid[x][y] = Particle(SOLID_MERCURY, x, y)
                changed = True
            elif particle.temperature >= 357:
                self.grid[x][y] = Particle(MERCURY_VAPOR, x, y)
                changed = True
        elif particle.type == SOLID_MERCURY and particle.temperature > -38.8:
            self.grid[x][y] = Particle(MERCURY, x, y)
            changed = True
        elif particle.type == SAND and particle.temperature >= 1600:
            self.grid[x][y] = Particle(LAVA, x, y)
            changed = True
        elif particle.type == GLASS and particle.temperature >= 1600:
            self.grid[x][y] = Particle(LAVA, x, y)
            changed = True

        # Particle interactions
        for dx, dy in NEIGHBORS:
            nx, ny = x + dx, y + dy
            if not self.in_bounds(nx, ny) or not self.grid[nx][ny]:
                continue
                
            neighbor = self.grid[nx][ny]
            key = (particle.type, neighbor.type)
            
            # Ice melts when touching hot particles
            if particle.type == ICE:
                if neighbor.type in (FIRE, LAVA, PLASMA) or neighbor.temperature > 50:
                    self.grid[x][y] = Particle(WATER, x, y)
                    changed = True
                    active_set.add((x, y))
                    continue
            
            if key in INTERACTION_CACHE:
                new_self, new_neighbor = INTERACTION_CACHE[key]
                if new_self:
                    self.grid[x][y] = Particle(new_self, x, y)
                    changed = True
                elif new_self is None:
                    self.grid[x][y] = None
                    self.all_particles.discard((x, y))
                    changed = True
                if new_neighbor:
                    self.grid[nx][ny] = Particle(new_neighbor, nx, ny)
                    changed = True
                elif new_neighbor is None:
                    self.grid[nx][ny] = None
                    self.all_particles.discard((nx, ny))
                    changed = True
            else:
                if particle.type == ACID:
                    if neighbor.type in (SAND, ICE, GLASS) and random.random() < 0.15:
                        self.grid[nx][ny] = None
                        self.all_particles.discard((nx, ny))
                        if random.random() < 0.1:
                            self.grid[nx][ny] = Particle(SMOKE, nx, ny)
                        changed = True
                elif particle.type == DILUTED_ACID:
                    if neighbor.type in (SAND, ICE) and random.random() < 0.05:
                        self.grid[nx][ny] = None
                        self.all_particles.discard((nx, ny))
                        changed = True
                elif particle.type == PLASMA and neighbor.type not in (CRYSTAL, VOID):
                    particle.temperature = max(5000, particle.temperature)
                    neighbor.temperature += 200
                    if random.random() < 0.05:
                        if neighbor.type.is_liquid:
                            self.grid[nx][ny] = Particle(STEAM, nx, ny)
                        else:
                            self.grid[nx][ny] = Particle(SMOKE, nx, ny)
                    changed = True
                elif particle.type == FIRE and neighbor.type.flammable:
                    if random.random() < 0.3:
                        self.grid[nx][ny] = Particle(FIRE, nx, ny)
                        if random.random() < 0.5:
                            above_y = ny - 1
                            if self.in_bounds(nx, above_y) and not self.grid[nx][above_y]:
                                self.grid[nx][above_y] = Particle(SMOKE, nx, above_y)
                        changed = True
                elif particle.type == VOID:
                    neighbor.temperature = max(0, neighbor.temperature - 50)

            if changed or (self.grid[nx][ny] and not self.grid[nx][ny].stationary):
                active_set.add((nx, ny))

        if particle.lifetime:
            particle.lifetime -= 1
            if particle.lifetime <= 0:
                if particle.type == FIRE and random.random() < 0.3:
                    self.grid[x][y] = Particle(SMOKE, x, y)
                else:
                    self.grid[x][y] = None
                    self.all_particles.discard((x, y))
                changed = True
                self.mark_active(x, y, active_set)

        if changed and self.grid[x][y]:
            self.grid[x][y].stationary = False

    def diffuse_temperature(self, active_set: Set[Tuple[int, int]]):
        new_temps = {}
        for x, y in active_set:
            if self.grid[x][y] and not self.grid[x][y].stationary:
                particle = self.grid[x][y]
                total, count = particle.temperature, 1
                for dx, dy in NEIGHBORS:
                    nx, ny = x + dx, y + dy
                    if self.in_bounds(nx, ny) and self.grid[nx][ny]:
                        neighbor = self.grid[nx][ny]
                        total += neighbor.temperature
                        count += 1
                avg_temp = total / count
                delta = (avg_temp - particle.temperature) * particle.type.thermal_conductivity
                if particle.type == FIRE:
                    delta += 50
                elif particle.type == PLASMA:
                    delta += 200
                elif particle.type == ICE:
                    delta -= 10
                elif particle.type == SOLID_MERCURY:
                    delta -= 5
                new_temps[(x, y)] = particle.temperature + delta
        for (x, y), temp in new_temps.items():
            if self.grid[x][y]:
                self.grid[x][y].temperature = temp
                self.grid[x][y].stationary = False

    def update(self):
        if not self.paused:
            if not self.active_set:
                self.active_set = self.all_particles.copy()
            for _ in range(self.simulation_speed):
                new_active = set()
                for x, y in list(self.active_set):
                    if self.grid[x][y] and not self.grid[x][y].updated:
                        self.update_particle(x, y, new_active)
                self.diffuse_temperature(self.active_set)
                for x, y in self.active_set:
                    if self.grid[x][y]:
                        self.grid[x][y].reset()
                self.active_set = new_active if new_active else self.active_set

    def draw(self, screen: pygame.Surface, fps: float):
        self.grid_surface.fill(tuple(BLACK))
        for x, y in self.all_particles:
            particle = self.grid[x][y]
            if particle:
                rect = pygame.Rect(x * self.cell_size, y * self.cell_size, 
                                 self.cell_size, self.cell_size)
                color = tuple(self._get_temp_color(particle) if self.show_temp else particle.color)
                pygame.draw.rect(self.grid_surface, color, rect)

        screen.blit(self.grid_surface, (0, 0))

        if self.show_grid:
            self._draw_grid(screen)
        self._draw_ui(screen, fps)

    def _get_temp_color(self, particle: Particle) -> array:
        temp_factor = min(1.0, max(-1.0, (particle.temperature - 20) / 280))
        if temp_factor < 0:
            return array('B', [0, 150, int(255 * (1 + temp_factor))])
        return array('B', [min(255, int(temp_factor * 255)), 150, max(0, int(255 * (1 - temp_factor)))])

    def _draw_grid(self, screen: pygame.Surface):
        for x in range(0, self.sim_width, self.cell_size):
            pygame.draw.line(screen, tuple(DARKGRAY), (x, 0), (x, self.height))
        for y in range(0, self.height, self.cell_size):
            pygame.draw.line(screen, tuple(DARKGRAY), (0, y), (self.sim_width, y))

    def _draw_ui(self, screen: pygame.Surface, fps: float):
        pygame.draw.rect(screen, tuple(UI_BG), (self.sim_width, 0, UI_WIDTH, self.height))
        title = self.title_font.render("Particle Sim", True, tuple(TITLE_COLOR))
        screen.blit(title, (self.sim_width + (UI_WIDTH - title.get_width()) // 2, 10))

        y_offset = 40
        slot_size = 30
        spacing = 10
        cols = 3
        
        hotbar_title = self.ui_font.render("Materials (1-9, Click others):", True, tuple(WHITE))
        screen.blit(hotbar_title, (self.sim_width + 10, y_offset))
        y_offset += 20

        for i, (p_type, rect) in enumerate(self.hotbar_slots):
            col = i % cols
            row = i // cols
            rect.x = self.sim_width + 10 + col * (slot_size + spacing + 50)
            rect.y = y_offset + row * (slot_size + spacing)
            border_color = tuple(SELECTED_COLOR if p_type == self.current_particle else GRAY)
            pygame.draw.rect(screen, border_color, rect, 2)
            pygame.draw.rect(screen, tuple(p_type.base_color), rect.inflate(-4, -4))
            label = self.ui_font.render(f"{i+1}. {p_type.name}" if i < 9 else p_type.name, True, tuple(WHITE))
            screen.blit(label, (rect.x + slot_size + 5, rect.y + (slot_size - label.get_height()) // 2))
        
        y_offset += 4 * (slot_size + spacing) + 20
        stats_title = self.ui_font.render("Simulation Info:", True, tuple(WHITE))
        screen.blit(stats_title, (self.sim_width + 10, y_offset))
        y_offset += 20
        
        stats = [
            f"FPS: {fps:.1f}",
            f"Brush Size: {self.brush_size} ([ / ])",
            f"Speed: {self.simulation_speed} (+/-)",
            "P: Pause",
            "G: Grid",
            "T: Temp",
            "C: Clear",
            "R: Reset",
            "",
            "Particle Counts:"
        ]
        for line in stats:
            text = self.ui_font.render(line, True, tuple(WHITE))
            screen.blit(text, (self.sim_width + 10, y_offset))
            y_offset += 15

        counts = {}
        total_temp, active_cells = 0, 0
        for x, y in self.all_particles:
            if self.grid[x][y]:
                p = self.grid[x][y]
                counts[p.type.name] = counts.get(p.type.name, 0) + 1
                total_temp += p.temperature
                active_cells += 1
        for name, count in sorted(counts.items()):
            text = self.ui_font.render(f"{name}: {count}", True, tuple(WHITE))
            screen.blit(text, (self.sim_width + 10, y_offset))
            y_offset += 15
        
        temp_text = f"Avg Temp: {total_temp/active_cells:.1f}°C" if active_cells else "Avg Temp: N/A"
        screen.blit(self.ui_font.render(temp_text, True, tuple(WHITE)), (self.sim_width + 10, y_offset))

    def handle_events(self, event: pygame.event.Event):
        if event.type == pygame.KEYDOWN:
            key_map = {
                pygame.K_p: lambda: setattr(self, 'paused', not self.paused),
                pygame.K_EQUALS: lambda: setattr(self, 'simulation_speed', self.simulation_speed + 1),
                pygame.K_MINUS: lambda: setattr(self, 'simulation_speed', max(1, self.simulation_speed - 1)),
                pygame.K_LEFTBRACKET: lambda: setattr(self, 'brush_size', max(1, self.brush_size - 1)),
                pygame.K_RIGHTBRACKET: lambda: setattr(self, 'brush_size', self.brush_size + 1),
                pygame.K_g: lambda: setattr(self, 'show_grid', not self.show_grid),
                pygame.K_t: lambda: setattr(self, 'show_temp', not self.show_temp),
                pygame.K_c: self.clear_grid,
                pygame.K_r: self.reset_boundaries,
                pygame.K_ESCAPE: lambda: pygame.event.post(pygame.event.Event(pygame.QUIT)),
                pygame.K_1: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[0]),
                pygame.K_2: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[1]),
                pygame.K_3: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[2]),
                pygame.K_4: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[3]),
                pygame.K_5: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[4]),
                pygame.K_6: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[5]),
                pygame.K_7: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[6]),
                pygame.K_8: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[7]),
                pygame.K_9: lambda: setattr(self, 'current_particle', PARTICLE_TYPES[8]),
            }
            if event.key in key_map:
                key_map[event.key]()

        if event.type == pygame.MOUSEBUTTONDOWN:
            mx, my = pygame.mouse.get_pos()
            if mx < self.sim_width:
                if event.button == 1:
                    self.mouse_held = True
                elif event.button == 3:
                    gx, gy = mx // self.cell_size, my // self.cell_size
                    self.apply_brush(gx, gy, False)
            else:
                if event.button == 1:
                    for p_type, rect in self.hotbar_slots:
                        if rect.collidepoint(mx, my):
                            self.current_particle = p_type
                            break
        elif event.type == pygame.MOUSEBUTTONUP:
            if event.button == 1:
                self.mouse_held = False
        elif event.type == pygame.MOUSEMOTION:
            mx, my = pygame.mouse.get_pos()
            if mx < self.sim_width and pygame.mouse.get_pressed()[2]:
                gx, gy = mx // self.cell_size, my // self.cell_size
                self.apply_brush(gx, gy, False)

    def update_mouse_input(self):
        if self.mouse_held:
            mx, my = pygame.mouse.get_pos()
            if mx < self.sim_width:
                gx, gy = mx // self.cell_size, my // self.cell_size
                self.apply_brush(gx, gy, True)

    def apply_brush(self, grid_x: int, grid_y: int, add: bool):
        for dx in range(-self.brush_size, self.brush_size + 1):
            for dy in range(-self.brush_size, self.brush_size + 1):
                if dx*dx + dy*dy <= self.brush_size * self.brush_size:
                    nx, ny = grid_x + dx, grid_y + dy
                    if self.in_bounds(nx, ny):
                        (self.add_particle if add else self.remove_particle)(nx, ny)

def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))
    pygame.display.set_caption("Enhanced Particle Simulation")
    clock = pygame.time.Clock()
    sim = Simulation(SIM_WIDTH, SCREEN_HEIGHT, CELL_SIZE)
    sim.reset_boundaries()

    running = True
    while running:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            sim.handle_events(event)
        
        sim.update_mouse_input()
        sim.update()
        sim.draw(screen, clock.get_fps())
        pygame.display.flip()
        clock.tick(FPS)
    pygame.quit()
    sys.exit()

if __name__ == "__main__":
    main()