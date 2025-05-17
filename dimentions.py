import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import itertools
import colorsys

# Settings
WIDTH, HEIGHT = 950, 620
max_dims = 12
dim = 3
angle = 0
rotation_speed = 0.01

slider_x, slider_y = 50, HEIGHT - 50
slider_w, slider_h = WIDTH - 100, 12
slider_handle_w = 16

pygame.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT), DOUBLEBUF | OPENGL)
pygame.display.set_caption("N-Dimensional Hypercube Visualizer")
font = pygame.font.SysFont("consolas", 20)

# Enable blending for UI overlays
glEnable(GL_BLEND)
glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)

# Global VBOs and related data
vbo_vertices = None
vbo_colors = None
vbo_points = None
total_lines = 0
num_points = 0
current_dim = -1

def get_hypercube_points(n):
    return np.array(list(itertools.product([-1, 1], repeat=n)), dtype=np.float32)

def dynamic_rotation_matrix(n, time):
    mat = np.eye(n, dtype=np.float32)
    idx = 0
    for i in range(n):
        for j in range(i + 1, n):
            speed = 0.4 + (idx % 7) * 0.2
            phase = idx * 0.5
            a = time * speed + np.sin(time * 0.3 + phase) * 0.5
            c, s = np.cos(a), np.sin(a)
            rot = np.eye(n, dtype=np.float32)
            rot[i, i] = c
            rot[j, j] = c
            rot[i, j] = -s
            rot[j, i] = s
            mat = mat @ rot
            idx += 1
    return mat

def wobble_matrix(n, time):
    mat = np.eye(n, dtype=np.float32)
    for i in range(n):
        scale = 1.0 + 0.25 * np.sin(time * (0.8 + i * 0.37))
        mat[i, i] = scale
    return mat

def project(points, n):
    if n >= 3:
        return points[:, :3]
    elif n == 2:
        return np.hstack((points, np.zeros((len(points), 1), dtype=np.float32)))
    else:
        pad = np.zeros((len(points), 3), dtype=np.float32)
        pad[:, :n] = points
        return pad

def draw_text(text, x, y):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    surf = font.render(text, True, (255, 255, 255))
    text_data = pygame.image.tostring(surf, "RGBA", True)
    glWindowPos2d(x, y)
    glDrawPixels(surf.get_width(), surf.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_slider(current_dim):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix(); glLoadIdentity()
    glOrtho(0, WIDTH, HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix(); glLoadIdentity()
    glDisable(GL_DEPTH_TEST)

    glColor4f(0.2, 0.2, 0.2, 0.7)
    glBegin(GL_QUADS)
    glVertex2f(slider_x, slider_y)
    glVertex2f(slider_x + slider_w, slider_y)
    glVertex2f(slider_x + slider_w, slider_y + slider_h)
    glVertex2f(slider_x, slider_y + slider_h)
    glEnd()

    ratio = (current_dim - 1) / (max_dims - 1)
    hx = slider_x + ratio * (slider_w - slider_handle_w)
    glColor4f(0.4, 0.8, 1.0, 0.9)
    glBegin(GL_QUADS)
    glVertex2f(hx, slider_y - 4)
    glVertex2f(hx + slider_handle_w, slider_y - 4)
    glVertex2f(hx + slider_handle_w, slider_y + slider_h + 4)
    glVertex2f(hx, slider_y + slider_h + 4)
    glEnd()

    glEnable(GL_DEPTH_TEST)
    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def setup_buffers(dim):
    global vbo_colors, total_lines, num_points
    num_points = 2 ** dim
    total_lines = num_points * (num_points - 1) // 2
    colors = np.zeros((2 * total_lines, 3), dtype=np.float32)
    count = 0
    for i in range(num_points):
        for j in range(i + 1, num_points):
            h = count / total_lines
            r, g, b = colorsys.hsv_to_rgb(h, 1.0, 1.0)
            colors[2 * count] = [r, g, b]
            colors[2 * count + 1] = [r, g, b]  # Same color for both vertices
            count += 1
    if vbo_colors is None:
        vbo_colors = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
    glBufferData(GL_ARRAY_BUFFER, colors.nbytes, colors, GL_STATIC_DRAW)

def update_vertex_buffer(projected):
    global vbo_vertices, total_lines
    vertex_array = np.zeros((2 * total_lines, 3), dtype=np.float32)
    count = 0
    for i in range(len(projected)):
        for j in range(i + 1, len(projected)):
            vertex_array[2 * count] = projected[i]
            vertex_array[2 * count + 1] = projected[j]
            count += 1
    if vbo_vertices is None:
        vbo_vertices = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glBufferData(GL_ARRAY_BUFFER, vertex_array.nbytes, vertex_array, GL_DYNAMIC_DRAW)

def setup_points_vbo(projected):
    global vbo_points
    if vbo_points is None:
        vbo_points = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_points)
    glBufferData(GL_ARRAY_BUFFER, projected.nbytes, projected, GL_DYNAMIC_DRAW)

def gl_draw_lines_vbo():
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_COLOR_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_vertices)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_colors)
    glColorPointer(3, GL_FLOAT, 0, None)
    glDrawArrays(GL_LINES, 0, 2 * total_lines)
    glDisableClientState(GL_COLOR_ARRAY)
    glDisableClientState(GL_VERTEX_ARRAY)

def gl_draw_points_vbo():
    glPointSize(6)
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_points)
    glVertexPointer(3, GL_FLOAT, 0, None)
    glColor3f(1.0, 0.5, 0.5)
    glDrawArrays(GL_POINTS, 0, num_points)
    glDisableClientState(GL_VERTEX_ARRAY)

def main():
    global dim, angle, current_dim
    clock = pygame.time.Clock()
    dragging = False
    running = True

    glEnable(GL_POINT_SMOOTH)
    glClearColor(0.05, 0.05, 0.1, 1)
    gluPerspective(40, WIDTH/HEIGHT, 0.1, 100.0)
    glTranslatef(0, 0, -10)
    setup_buffers(dim)  # Initial setup
    current_dim = dim

    while running:
        for e in pygame.event.get():
            if e.type == QUIT:
                running = False
            elif e.type == MOUSEBUTTONDOWN:
                mx, my = e.pos
                if slider_y - 10 <= my <= slider_y + slider_h + 10:
                    dragging = True
            elif e.type == MOUSEBUTTONUP:
                dragging = False
            elif e.type == KEYDOWN:
                if e.key == K_UP:
                    dim = min(dim + 1, max_dims)
                elif e.key == K_DOWN:
                    dim = max(dim - 1, 1)

        if dragging:
            mx = pygame.mouse.get_pos()[0]
            ratio = (mx - slider_x) / (slider_w - slider_handle_w)
            dim = int(np.clip(round(ratio * (max_dims-1) + 1), 1, max_dims))

        if dim != current_dim:
            setup_buffers(dim)
            current_dim = dim

        pts_nd = get_hypercube_points(dim)
        rot = dynamic_rotation_matrix(dim, angle)
        wobble = wobble_matrix(dim, angle)
        transformed = pts_nd @ rot @ wobble
        projected = project(transformed, dim)

        update_vertex_buffer(projected)
        setup_points_vbo(projected)

        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glPushMatrix()
        glRotatef(angle * 20, 1, 1, 0)
        gl_draw_lines_vbo()
        gl_draw_points_vbo()
        glPopMatrix()

        draw_slider(dim)
        draw_text(f"{dim}D  ({2**dim} verts)", 10, 10)
        draw_text(f"FPS: {int(clock.get_fps())}", WIDTH - 100, 10)

        pygame.display.flip()
        clock.tick(60)
        angle += rotation_speed

    # Cleanup
    if vbo_vertices:
        glDeleteBuffers(1, [vbo_vertices])
    if vbo_colors:
        glDeleteBuffers(1, [vbo_colors])
    if vbo_points:
        glDeleteBuffers(1, [vbo_points])
    pygame.quit()

if __name__ == "__main__":
    main()