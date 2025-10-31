"""
Advanced aesthetic & performant particle sim
Controls:
  LMB hold + drag : spawn particles along the drag (line emission)
  RMB hold        : apply mouse force (attract / repel dependent on mode)
  Mouse wheel     : change emission spacing (hold slower/faster line density)
  SPACE           : pause / resume
  C               : clear particles
  G               : toggle gravity
  V               : toggle collisions
  T               : toggle trails
  H               : toggle glow (bloom)
  Q               : toggle quality (low / high) - disables trails/glow
  [ / ]           : decrease / increase particle radius
  - / =           : decrease / increase emission rate (max particles per sec)
  M               : cycle mouse force mode (none -> attract -> repel)
  S               : save screenshot (screenshot_TIMESTAMP.png)
  ESC             : quit
"""

import pygame, random, math, colorsys, time, os
from collections import defaultdict

pygame.init()
os.environ['SDL_VIDEO_CENTERED'] = '1'

# ---------- Configurable constants ----------
WIDTH, HEIGHT = 1100, 700
TARGET_FPS = 60

# Physics (units: pixels, seconds)
DEFAULT_RADIUS = 5
MIN_RADIUS = 2
MAX_RADIUS = 18

GRAVITY = 1200.0          # px / s^2
AIR_RESIST_BASE = 0.995   # per-frame style base (converted with dt)
ELASTICITY = 0.85         # restitution for wall collisions and collisions between particles

# Emission
MAX_PARTICLES = 2500
EMIT_RATE = 120.0          # particles per second when stationary (line mode depends on spacing)
EMIT_SPACING = 2.2         # times radius (spacing between spawned particles on the line)

# Grid (spatial hash)
GRID_CELL = 24  # will be recomputed in setup to be radius * factor

# Aesthetics
BG_TOP = (12, 14, 30)
BG_BOTTOM = (30, 28, 50)
TRAIL_ALPHA = 18        # how strongly new draws are applied to trail surface
TRAIL_FADE = 10         # how fast trail surface fades (0 = no fade, larger = faster)
GLOW_SCALE = 2.8        # glow radius multiplier
MAX_COLOR_SPEED = 900   # speed used to map hue by velocity

# ---------- Setup ----------
screen = pygame.display.set_mode((WIDTH, HEIGHT))
pygame.display.set_caption("Particle Garden — LMB drag to paint in lines")
font = pygame.font.Font(None, 20)
clock = pygame.time.Clock()

# Surfaces for effects
trail_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
glow_surf = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)

# ---------- Utilities ----------
def hsv_to_rgb(h, s, v):
    r, g, b = colorsys.hsv_to_rgb(h, s, v)
    return int(r * 255), int(g * 255), int(b * 255)

def draw_vertical_gradient(surf, top_color, bottom_color):
    """Quick vertical gradient fill"""
    w, h = surf.get_size()
    for y in range(h):
        t = y / (h - 1)
        r = int(top_color[0] * (1 - t) + bottom_color[0] * t)
        g = int(top_color[1] * (1 - t) + bottom_color[1] * t)
        b = int(top_color[2] * (1 - t) + bottom_color[2] * t)
        surf.fill((r, g, b), rect=pygame.Rect(0, y, w, 1))

# ---------- Particle class ----------
class Particle:
    __slots__ = ("x", "y", "vx", "vy", "radius", "mass", "neighbors", "idx")

    def __init__(self, x, y, radius=DEFAULT_RADIUS):
        self.x = float(x)
        self.y = float(y)
        # initialize velocities in px/s (so we multiply by dt when integrating)
        self.vx = random.uniform(-160, 160)
        self.vy = random.uniform(-160, 160)
        self.radius = radius
        self.mass = radius * radius  # simple mass ~ area
        self.neighbors = 0
        self.idx = 0

    def integrate(self, dt, drag_factor, gravity_on):
        # drag
        self.vx *= drag_factor
        self.vy *= drag_factor
        # gravity in y
        if gravity_on:
            self.vy += GRAVITY * dt
        # integrate
        self.x += self.vx * dt
        self.y += self.vy * dt
        # wall collisions (simple reflection with restitution)
        r = self.radius
        if self.x < r:
            self.x = r
            self.vx = -self.vx * ELASTICITY
        elif self.x > WIDTH - r:
            self.x = WIDTH - r
            self.vx = -self.vx * ELASTICITY
        if self.y < r:
            self.y = r
            self.vy = -self.vy * ELASTICITY
        elif self.y > HEIGHT - r:
            self.y = HEIGHT - r
            self.vy = -self.vy * ELASTICITY

    def draw_main(self, surf, color):
        pygame.draw.circle(surf, color, (int(self.x), int(self.y)), int(self.radius))

    def draw_trail(self, surf, color_with_alpha):
        pygame.draw.circle(surf, color_with_alpha, (int(self.x), int(self.y)), int(self.radius))

    def draw_glow(self, surf, color_with_alpha, glow_radius):
        pygame.draw.circle(surf, color_with_alpha, (int(self.x), int(self.y)), int(glow_radius))


# ---------- Simulation state ----------
particles = []
running = True
paused = False
gravity_on = True
collisions_on = True
trails_on = True
glow_on = True
low_quality = False

emit_rate = EMIT_RATE  # particles per second (when stationary)
emit_spacing_factor = EMIT_SPACING  # times radius

mouse_force_mode = 1  # 0 none, 1 attract, 2 repel
mouse_force_strength = 4_000_000.0  # tune; used in inverse-square * mass
last_mouse_pos = None
last_add_pos = None
add_timer = 0.0

# recompute grid cell size to be proportional to radius
GRID_CELL = int(DEFAULT_RADIUS * 3.7)
GRID_CELL = max(18, GRID_CELL)

# ---------- Collision resolution ----------
def resolve_collisions(particles, grid, cell_size, restitution):
    # reset neighbor counts
    for p in particles:
        p.neighbors = 0

    for p in particles:
        gx = int(p.x // cell_size)
        gy = int(p.y // cell_size)
        # check neighbors in neighboring cells
        for dx in (-1, 0, 1):
            for dy in (-1, 0, 1):
                key = (gx + dx, gy + dy)
                if key not in grid: 
                    continue
                for q in grid[key]:
                    # avoid double-handling pairs:
                    if q.idx <= p.idx:
                        continue
                    # overlap test
                    rx = q.x - p.x
                    ry = q.y - p.y
                    dist2 = rx * rx + ry * ry
                    min_dist = p.radius + q.radius
                    if dist2 < min_dist * min_dist and dist2 > 0.0001:
                        dist = math.sqrt(dist2)
                        nx = rx / dist
                        ny = ry / dist
                        # position correction (mass proportional)
                        overlap = min_dist - dist
                        # distribute correction based on mass (prevents tunneling)
                        inv_mass_sum = 1.0 / (p.mass + q.mass)
                        p.x -= nx * overlap * (q.mass * inv_mass_sum)
                        p.y -= ny * overlap * (q.mass * inv_mass_sum)
                        q.x += nx * overlap * (p.mass * inv_mass_sum)
                        q.y += ny * overlap * (p.mass * inv_mass_sum)

                        # velocity impulse (1D along normal)
                        rel_vel = (p.vx - q.vx) * nx + (p.vy - q.vy) * ny
                        if rel_vel < 0:  # only if they are moving into each other
                            j = -(1 + restitution) * rel_vel
                            j /= (1.0 / p.mass + 1.0 / q.mass)
                            # apply impulses
                            p.vx += (j * nx) / p.mass
                            p.vy += (j * ny) / p.mass
                            q.vx -= (j * nx) / q.mass
                            q.vy -= (j * ny) / q.mass

                        # neighbor count for coloring
                        p.neighbors += 1
                        q.neighbors += 1

# ---------- Emission helpers ----------
def spawn_particle_at(x, y, radius_override=None):
    r = radius_override if radius_override is not None else DEFAULT_RADIUS
    if len(particles) < MAX_PARTICLES:
        particles.append(Particle(x, y, radius_override or DEFAULT_RADIUS))

def spawn_along_line(prev_pos, cur_pos, spacing_px, radius_override=None):
    """Spawn particles along segment from prev_pos to cur_pos with spacing spacing_px.
       Returns new last spawn position."""
    if prev_pos is None:
        return cur_pos
    px, py = prev_pos
    cx, cy = cur_pos
    dx = cx - px
    dy = cy - py
    dist = math.hypot(dx, dy)
    if dist < 0.0001:
        # stationary — spawn single particle (rate-limited elsewhere)
        return prev_pos
    nx = dx / dist
    ny = dy / dist
    # move along the segment spawning at spacing
    # we move prev_pos forward each time we spawn
    while dist >= spacing_px and len(particles) < MAX_PARTICLES:
        px += nx * spacing_px
        py += ny * spacing_px
        spawn_particle_at(px + random.uniform(-0.5, 0.5), py + random.uniform(-0.5, 0.5), radius_override)
        dx = cx - px
        dy = cy - py
        dist = math.hypot(dx, dy)
    return (px, py)

# ---------- Drawing UI ----------
def draw_ui(surf, fps, values):
    lines = [
        f"Particles: {len(particles):,} / {MAX_PARTICLES}   FPS: {fps}   Quality: {'LOW' if low_quality else 'HIGH'}",
        f"Emit spacing (mouse wheel): {emit_spacing_factor:.2f}×radius   Emit/sec (stationary): {emit_rate:.0f}",
        f"Radius [ / ]: {DEFAULT_RADIUS}   Gravity (G): {'ON' if gravity_on else 'OFF'}   Collisions (V): {'ON' if collisions_on else 'OFF'}",
        f"Trails (T): {'ON' if trails_on else 'OFF'}   Glow (H): {'ON' if glow_on else 'OFF'}",
        f"Mouse force (M): {['NONE','ATTRACT','REPEL'][mouse_force_mode]}   Strength +/-: mouse_force_strength={int(mouse_force_strength)}",
        "Controls: LMB-drag emit | RMB-hold apply mouse force | SPACE pause | C clear | Q quality toggle | S save screenshot"
    ]
    x = 12
    y = 10
    for i, ln in enumerate(lines):
        txt = font.render(ln, True, (200, 200, 200))
        surf.blit(txt, (x, y + i * 18))

# ---------- Main loop ----------
def main():
    global paused, gravity_on, collisions_on, trails_on, glow_on, low_quality
    global emit_rate, emit_spacing_factor, DEFAULT_RADIUS, GRID_CELL
    global mouse_force_mode, mouse_force_strength
    global last_mouse_pos, last_add_pos, add_timer

    # pre-rendered background gradient (static)
    bg = pygame.Surface((WIDTH, HEIGHT))
    draw_vertical_gradient(bg, BG_TOP, BG_BOTTOM)

    last_time = time.time()

    while True:
        dt = clock.tick(TARGET_FPS) / 1000.0
        fps = int(clock.get_fps())
        for ev in pygame.event.get():
            if ev.type == pygame.QUIT:
                pygame.quit()
                return
            elif ev.type == pygame.KEYDOWN:
                if ev.key == pygame.K_ESCAPE:
                    pygame.quit()
                    return
                elif ev.key == pygame.K_SPACE:
                    paused = not paused
                elif ev.key == pygame.K_c:
                    particles.clear()
                    trail_surf.fill((0, 0, 0, 0))
                elif ev.key == pygame.K_g:
                    gravity_on = not gravity_on
                elif ev.key == pygame.K_v:
                    collisions_on = not collisions_on
                elif ev.key == pygame.K_t:
                    trails_on = not trails_on
                elif ev.key == pygame.K_h:
                    glow_on = not glow_on
                elif ev.key == pygame.K_q:
                    low_quality = not low_quality
                elif ev.key == pygame.K_LEFTBRACKET:
                    DEFAULT_RADIUS = max(MIN_RADIUS, DEFAULT_RADIUS - 1)
                    GRID_CELL = max(18, int(DEFAULT_RADIUS * 3.7))
                elif ev.key == pygame.K_RIGHTBRACKET:
                    DEFAULT_RADIUS = min(MAX_RADIUS, DEFAULT_RADIUS + 1)
                    GRID_CELL = max(18, int(DEFAULT_RADIUS * 3.7))
                elif ev.key == pygame.K_MINUS or ev.key == pygame.K_KP_MINUS:
                    emit_rate = max(1.0, emit_rate - 10.0)
                elif ev.key == pygame.K_EQUALS or ev.key == pygame.K_PLUS:
                    emit_rate = min(2000.0, emit_rate + 10.0)
                elif ev.key == pygame.K_m:
                    mouse_force_mode = (mouse_force_mode + 1) % 3
                elif ev.key == pygame.K_s:
                    pygame.image.save(screen, f"screenshot_{int(time.time())}.png")
            elif ev.type == pygame.MOUSEBUTTONDOWN:
                if ev.button == 1:
                    last_add_pos = ev.pos
                elif ev.button == 3:
                    # right click starts mouse force application (handled below)
                    last_mouse_pos = ev.pos
                elif ev.button == 4:  # wheel up -> tighter spacing
                    emit_spacing_factor = max(0.3, emit_spacing_factor - 0.1)
                elif ev.button == 5:  # wheel down -> sparser spacing
                    emit_spacing_factor = min(8.0, emit_spacing_factor + 0.1)
            elif ev.type == pygame.MOUSEBUTTONUP:
                if ev.button == 1:
                    last_add_pos = None
                elif ev.button == 3:
                    last_mouse_pos = None
            elif ev.type == pygame.MOUSEMOTION:
                # track last mouse for spawning lines
                pass

        mx, my = pygame.mouse.get_pos()
        mbuttons = pygame.mouse.get_pressed()

        # emission handling (LMB drag line)
        spacing_px = emit_spacing_factor * max(2.0, DEFAULT_RADIUS * 2.0)
        if mbuttons[0]:  # left held
            if last_add_pos is None:
                last_add_pos = (mx, my)
            # spawn along line from last_add_pos to current mouse
            last_add_pos = spawn_along_line(last_add_pos, (mx, my), spacing_px, radius_override=DEFAULT_RADIUS)
            # if stationary (no movement) we also want to spawn at a rate:
            # use add_timer to spawn based on emit_rate
            if abs(mx - last_add_pos[0]) < 1.0 and abs(my - last_add_pos[1]) < 1.0:
                add_timer += dt
                per_particle_time = 1.0 / max(1.0, emit_rate / 10.0)
                while add_timer >= per_particle_time and len(particles) < MAX_PARTICLES:
                    spawn_particle_at(mx + random.uniform(-1.0, 1.0), my + random.uniform(-1.0, 1.0), DEFAULT_RADIUS)
                    add_timer -= per_particle_time
        else:
            last_add_pos = None
            add_timer = 0.0

        # mouse force (RMB)
        mouse_force_active = mbuttons[2]

        # integrate step (apply forces & integrate)
        if not paused:
            # compute drag factor from base; convert per-frame-style base to dt-scaled
            # we want AIR_RESIST_BASE to behave like a per-frame factor near 60 FPS
            drag_factor = pow(AIR_RESIST_BASE, dt * TARGET_FPS)

            # apply attraction/repel to particles if RMB held
            if mouse_force_active and mouse_force_mode != 0:
                for p in particles:
                    rx = mx - p.x
                    ry = my - p.y
                    dist2 = rx * rx + ry * ry + 1e-6
                    # inverse-square with attenuation
                    force = mouse_force_strength / dist2
                    # direction
                    ds = math.sqrt(dist2)
                    nx = rx / ds
                    ny = ry / ds
                    if mouse_force_mode == 1:
                        # attract
                        ax = nx * force / p.mass
                        ay = ny * force / p.mass
                    else:
                        # repel
                        ax = -nx * force / p.mass
                        ay = -ny * force / p.mass
                    # integrate half-step manually (add immediate acceleration)
                    p.vx += ax * dt
                    p.vy += ay * dt

            # integrate movement
            for p in particles:
                p.integrate(dt, drag_factor, gravity_on)

            # assign idx for pair deduplication
            for i, p in enumerate(particles):
                p.idx = i

            # build spatial hash grid
            grid = defaultdict(list)
            cell_size = GRID_CELL
            for p in particles:
                key = (int(p.x // cell_size), int(p.y // cell_size))
                grid[key].append(p)

            # collisions
            if collisions_on and not low_quality:
                resolve_collisions(particles, grid, cell_size, ELASTICITY)
            elif collisions_on:  # low quality collisions still somewhat checked but cheaper
                # only check pairs inside same cell (not neighbors) to reduce cost
                for cell_list in grid.values():
                    ln = len(cell_list)
                    for i in range(ln):
                        p = cell_list[i]
                        for j in range(i + 1, ln):
                            q = cell_list[j]
                            rx = q.x - p.x
                            ry = q.y - p.y
                            dist2 = rx * rx + ry * ry
                            min_dist = p.radius + q.radius
                            if dist2 < min_dist * min_dist and dist2 > 0.0001:
                                dist = math.sqrt(dist2)
                                nx = rx / dist
                                ny = ry / dist
                                overlap = min_dist - dist
                                inv_mass_sum = 1.0 / (p.mass + q.mass)
                                p.x -= nx * overlap * (q.mass * inv_mass_sum)
                                p.y -= ny * overlap * (q.mass * inv_mass_sum)
                                q.x += nx * overlap * (p.mass * inv_mass_sum)
                                q.y += ny * overlap * (p.mass * inv_mass_sum)
                                rel_vel = (p.vx - q.vx) * nx + (p.vy - q.vy) * ny
                                if rel_vel < 0:
                                    j = -(1 + ELASTICITY) * rel_vel
                                    j /= (1.0 / p.mass + 1.0 / q.mass)
                                    p.vx += (j * nx) / p.mass
                                    p.vy += (j * ny) / p.mass
                                    q.vx -= (j * nx) / q.mass
                                    q.vy -= (j * ny) / q.mass

        # ----- Rendering -----
        # background gradient
        screen.blit(bg, (0, 0))

        # trails handling
        if trails_on and not low_quality:
            # fade trails surface gradually
            fade_rect = pygame.Surface((WIDTH, HEIGHT), pygame.SRCALPHA)
            fade_rect.fill((0, 0, 0, TRAIL_FADE))
            trail_surf.blit(fade_rect, (0, 0))
        else:
            # clear if trails off or low quality
            trail_surf.fill((0, 0, 0, 0))

        # glow surface clear each frame
        glow_surf.fill((0, 0, 0, 0))

        # prepare drawing: build grid again for coloring density if needed
        # (cheap enough to reuse previous grid in many cases, but recompute here for correctness)
        grid_for_color = defaultdict(int)
        if not paused:
            for p in particles:
                key = (int(p.x // GRID_CELL), int(p.y // GRID_CELL))
                grid_for_color[key] += 1

        # draw particles
        for p in particles:
            # color modes (speed / density)
            speed = math.hypot(p.vx, p.vy)
            # map hue from blue (slow) to red (fast)
            h = max(0.0, 0.66 - min(speed / MAX_COLOR_SPEED, 1.0) * 0.66)
            # brightness influenced by neighbor count too
            density_factor = min(1.0, p.neighbors / 4.0)
            v = 0.9 - 0.25 * density_factor  # slightly dimmer when denser
            s = 0.9
            rgb = hsv_to_rgb(h, s, v)
            # draw onto trail surface (soft)
            if trails_on and not low_quality:
                trail_color = (rgb[0], rgb[1], rgb[2], TRAIL_ALPHA)
                p.draw_trail(trail_surf, trail_color)
            # draw glow
            if glow_on and not low_quality:
                glow_color = (rgb[0], rgb[1], rgb[2], 18)
                glow_r = p.radius * GLOW_SCALE
                p.draw_glow(glow_surf, glow_color, glow_r)
            # main crisp circle
            p.draw_main(screen, rgb)

        # composite glow using additive blending
        if glow_on and not low_quality:
            screen.blit(glow_surf, (0, 0), special_flags=pygame.BLEND_ADD)
        # composite trails (below the crisp circles so trails appear behind)
        if trails_on and not low_quality:
            screen.blit(trail_surf, (0, 0))

        # UI
        draw_ui(screen, fps, None)

        # mode indicator small boxes
        mode_text = f"[SPACE] Pause: {'PAUSED' if paused else 'RUNNING'}"
        screen.blit(font.render(mode_text, True, (200, 200, 200)), (12, HEIGHT - 26))

        pygame.display.flip()

# run
if __name__ == "__main__":
    main()
