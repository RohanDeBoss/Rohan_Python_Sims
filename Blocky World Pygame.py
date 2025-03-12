import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import random
import threading
import math
import ctypes
from opensimplex import OpenSimplex
from concurrent.futures import ThreadPoolExecutor


# --- Constants ---
SCREEN_WIDTH = 1600
SCREEN_HEIGHT = 900
MOVE_SPEED = 1.0  # Player movement speed
ROTATION_SPEED = 0.9  # Mouse sensitivity
CHUNK_SIZE = 16  # Chunk size in blocks
VIEW_DISTANCE = 16  # Render distance in chunks
SEED = random.randint(0, 100000)  # Seed for procedural generation

# Global thread pool for asynchronous mesh generation.
executor = ThreadPoolExecutor(max_workers=6)

# --- Noise Generation ---
noise_generator = OpenSimplex(seed=SEED)

def get_height(x, z):
    scale1 = 0.05
    scale2 = 0.01
    height = noise_generator.noise2(x * scale1, z * scale1) * 5
    height += noise_generator.noise2(x * scale2, z * scale2) * 20
    return height

def get_biome(x, z):
    scale = 0.02
    biome_noise = noise_generator.noise2(x * scale, z * scale)
    if biome_noise < -0.3:
        return "water"
    elif biome_noise < 0.2:
        return "grass"
    elif biome_noise < 0.6:
        return "forest"
    else:
        return "mountain"

def extract_frustum_planes(proj, modl):
    clip = np.dot(proj, modl).astype(np.float32)
    planes = np.zeros((6, 4), dtype=np.float32)
    # Left
    planes[0][0] = clip[0][3] + clip[0][0]
    planes[0][1] = clip[1][3] + clip[1][0]
    planes[0][2] = clip[2][3] + clip[2][0]
    planes[0][3] = clip[3][3] + clip[3][0]
    # Right
    planes[1][0] = clip[0][3] - clip[0][0]
    planes[1][1] = clip[1][3] - clip[1][0]
    planes[1][2] = clip[2][3] - clip[2][0]
    planes[1][3] = clip[3][3] - clip[3][0]
    # Bottom
    planes[2][0] = clip[0][3] + clip[0][1]
    planes[2][1] = clip[1][3] + clip[1][1]
    planes[2][2] = clip[2][3] + clip[2][1]
    planes[2][3] = clip[3][3] + clip[3][1]
    # Top
    planes[3][0] = clip[0][3] - clip[0][1]
    planes[3][1] = clip[1][3] - clip[1][1]
    planes[3][2] = clip[2][3] - clip[2][1]
    planes[3][3] = clip[3][3] - clip[3][1]
    # Near
    planes[4][0] = clip[0][3] + clip[0][2]
    planes[4][1] = clip[1][3] + clip[1][2]
    planes[4][2] = clip[2][3] + clip[2][2]
    planes[4][3] = clip[3][3] + clip[3][2]
    # Far
    planes[5][0] = clip[0][3] - clip[0][2]
    planes[5][1] = clip[1][3] - clip[1][2]
    planes[5][2] = clip[2][3] - clip[2][2]
    planes[5][3] = clip[3][3] - clip[3][2]

    for i in range(6):
        length = math.sqrt(planes[i][0]**2 + planes[i][1]**2 + planes[i][2]**2)
        planes[i] /= length

    return planes

def is_chunk_in_frustum(chunk_x, chunk_z, planes):
    min_x = chunk_x * CHUNK_SIZE
    max_x = min_x + CHUNK_SIZE
    min_y = 0  # Minimum height
    max_y = 255  # Maximum height
    min_z = chunk_z * CHUNK_SIZE
    max_z = min_z + CHUNK_SIZE
    
    for plane in planes:
        px = max_x if plane[0] > 0 else min_x
        py = max_y if plane[1] > 0 else min_y
        pz = max_z if plane[2] > 0 else min_z
        
        positive_dist = plane[0] * px + plane[1] * py + plane[2] * pz + plane[3]
        if positive_dist < 0:
            return False
    return True

# --- Improved Lighting ---
def setup_lighting():
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glLightfv(GL_LIGHT0, GL_POSITION, (-2, 4, -2, 0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, (0.2, 0.2, 0.2, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, (0.9, 0.9, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, (0.5, 0.5, 0.5, 1.0))
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, (0.3, 0.3, 0.3, 1.0))
    glEnable(GL_NORMALIZE)
    glShadeModel(GL_SMOOTH)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)

class Chunk:
    # Predefined constants for efficiency
    TEX_COORDS = {
        "grass": [(0, 1), (1, 1), (1, 0), (0, 0)],      # Atlas coords for grass
        "dirt": [(2, 0), (3, 0), (3, 1), (2, 1)],       # Atlas coords for dirt
        "stone": [(1, 0), (2, 0), (2, 1), (1, 1)],      # Atlas coords for stone
        "log": [(4, 1), (5, 1), (5, 0), (4, 0)],        # Atlas coords for log
        "leaves": [(0, 0), (1, 0), (1, 1), (0, 1)],     # Full texture for leaves
        "water": [(0, 0), (1, 0), (1, 1), (0, 1)],      # Full texture for water
        "sand": [(0, 0), (1, 0), (1, 1), (0, 1)],       # Full texture for sand
    }
    FACE_VERTICES = {
        "top": [(0, 1, 0), (1, 1, 0), (1, 1, 1), (0, 1, 1)],
        "bottom": [(0, 0, 0), (1, 0, 0), (1, 0, 1), (0, 0, 1)],
        "front": [(0, 0, 0), (1, 0, 0), (1, 1, 0), (0, 1, 0)],
        "back": [(0, 0, 1), (1, 0, 1), (1, 1, 1), (0, 1, 1)],
        "left": [(0, 0, 0), (0, 0, 1), (0, 1, 1), (0, 1, 0)],
        "right": [(1, 0, 0), (1, 0, 1), (1, 1, 1), (1, 1, 0)]
    }
    NORMALS = {
        "top": (0, 1, 0), "bottom": (0, -1, 0), "front": (0, 0, -1),
        "back": (0, 0, 1), "left": (-1, 0, 0), "right": (1, 0, 0)
    }
    FACE_DIRECTIONS = [
        ("top", (0, 1, 0)), ("bottom", (0, -1, 0)), ("front", (0, 0, -1)),
        ("back", (0, 0, 1)), ("left", (-1, 0, 0)), ("right", (1, 0, 0))
    ]

    def __init__(self, x, z):
        self.x = x
        self.z = z
        self.blocks = {}
        self.generate()
        self.vbo_id = glGenBuffers(1)
        self.water_vbo_id = glGenBuffers(1)
        self.leaves_vbo_id = glGenBuffers(1)
        self.sand_vbo_id = glGenBuffers(1)  # New VBO for sand
        self.terrain_data = None
        self.water_data = None
        self.leaves_data = None
        self.sand_data = None
        self.num_vertices = 0
        self.num_water_vertices = 0
        self.num_leaves_vertices = 0
        self.num_sand_vertices = 0
        self.is_dirty = True
        self.future = executor.submit(self.generate_mesh_async)

    def set_water_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, (0.1, 0.2, 0.4, 0.6))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.2, 0.4, 0.8, 0.6))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.9, 0.9, 1.0, 0.7))
        glMaterialf(GL_FRONT, GL_SHININESS, 96.0)

    def set_leaves_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, (0.1, 0.3, 0.1, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.2, 0.6, 0.2, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.3, 0.5, 0.3, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 32.0)

    def set_sand_material(self):
        glMaterialfv(GL_FRONT, GL_AMBIENT, (0.7, 0.6, 0.3, 1.0))
        glMaterialfv(GL_FRONT, GL_DIFFUSE, (0.9, 0.8, 0.5, 1.0))
        glMaterialfv(GL_FRONT, GL_SPECULAR, (0.2, 0.2, 0.1, 1.0))
        glMaterialf(GL_FRONT, GL_SHININESS, 10.0)

    def generate(self):
        for x in range(CHUNK_SIZE):
            for z in range(CHUNK_SIZE):
                world_x = self.x * CHUNK_SIZE + x
                world_z = self.z * CHUNK_SIZE + z
                height = int(get_height(world_x, world_z))
                biome = get_biome(world_x, world_z)

                for y in range(height - 3, height + 1):
                    if y < height - 1:
                        self.blocks[(x, y, z)] = "stone"
                    elif y < height:
                        self.blocks[(x, y, z)] = "dirt"
                    else:
                        if biome in ("grass", "forest"):
                            self.blocks[(x, y, z)] = "grass"
                        elif biome == "water":
                            self.blocks[(x, y, z)] = "water"
                        elif biome == "mountain":
                            if random.random() < 0.5:  # 50% chance for sand on mountain surface
                                self.blocks[(x, y, z)] = "sand"
                            else:
                                self.blocks[(x, y, z)] = "stone"

                if biome == "forest" and random.random() < 0.1 and height > 0:
                    tree_height = random.randint(4, 7)
                    for y in range(height + 1, height + tree_height):
                        self.blocks[(x, y, z)] = "log"
                    for ly in range(height + tree_height - 2, height + tree_height + 1):
                        for lx in range(max(x - 2, 0), min(x + 3, CHUNK_SIZE)):
                            for lz in range(max(z - 2, 0), min(z + 3, CHUNK_SIZE)):
                                if (lx, ly, lz) not in self.blocks:
                                    self.blocks[(lx, ly, lz)] = "leaves"

    def generate_mesh_async(self):
        terrain_data = []
        water_data = []
        leaves_data = []
        sand_data = []
        blocks = self.blocks

        def add_face(x, y, z, face_type, block_type, data_list):
            norm = (0, 1, 0) if block_type == "water" else self.NORMALS[face_type]
            vertices = self.FACE_VERTICES[face_type]
            tex_coords = self.TEX_COORDS[block_type]
            for i in range(4):
                vx, vy, vz = vertices[i]
                tx, ty = tex_coords[i]
                data_list.extend([x + vx, y + vy, z + vz, tx, ty, norm[0], norm[1], norm[2]])

        for (x, y, z), block_type in blocks.items():
            if block_type == "water":
                if (x, y + 1, z) not in blocks:
                    add_face(x, y, z, "top", "water", water_data)
            if block_type != "air":
                for face_type, (dx, dy, dz) in self.FACE_DIRECTIONS:
                    neighbor = (x + dx, y + dy, z + dz)
                    neighbor_block = blocks.get(neighbor)
                    if neighbor_block is None or (block_type != "water" and neighbor_block == "water") or \
                       (block_type == "water" and neighbor_block != "water"):
                        if block_type == "water":
                            add_face(x, y, z, face_type, block_type, water_data)
                        elif block_type == "leaves":
                            add_face(x, y, z, face_type, block_type, leaves_data)
                        elif block_type == "sand":
                            add_face(x, y, z, face_type, block_type, sand_data)
                        else:
                            add_face(x, y, z, face_type, block_type, terrain_data)

        self.terrain_data = np.array(terrain_data, dtype=np.float32)
        self.water_data = np.array(water_data, dtype=np.float32)
        self.leaves_data = np.array(leaves_data, dtype=np.float32)
        self.sand_data = np.array(sand_data, dtype=np.float32)
        self.num_vertices = len(terrain_data) // 8
        self.num_water_vertices = len(water_data) // 8
        self.num_leaves_vertices = len(leaves_data) // 8
        self.num_sand_vertices = len(sand_data) // 8

    def upload_mesh(self):
        if self.terrain_data is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
            glBufferData(GL_ARRAY_BUFFER, self.terrain_data.nbytes, self.terrain_data, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        if self.water_data is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.water_vbo_id)
            if self.num_water_vertices > 0:
                glBufferData(GL_ARRAY_BUFFER, self.water_data.nbytes, self.water_data, GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        if self.leaves_data is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.leaves_vbo_id)
            if self.num_leaves_vertices > 0:
                glBufferData(GL_ARRAY_BUFFER, self.leaves_data.nbytes, self.leaves_data, GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        if self.sand_data is not None:
            glBindBuffer(GL_ARRAY_BUFFER, self.sand_vbo_id)
            if self.num_sand_vertices > 0:
                glBufferData(GL_ARRAY_BUFFER, self.sand_data.nbytes, self.sand_data, GL_STATIC_DRAW)
            else:
                glBufferData(GL_ARRAY_BUFFER, 0, None, GL_STATIC_DRAW)
            glBindBuffer(GL_ARRAY_BUFFER, 0)
        self.is_dirty = False

    def check_mesh_update(self):
        if self.is_dirty and self.future.done():
            self.upload_mesh()

    def render(self):
        self.check_mesh_update()
        if self.is_dirty or self.num_vertices == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_id)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 8 * 4, ctypes.c_void_p(0))
        glTexCoordPointer(2, GL_FLOAT, 8 * 4, ctypes.c_void_p(3 * 4))
        glNormalPointer(GL_FLOAT, 8 * 4, ctypes.c_void_p(5 * 4))
        glDrawArrays(GL_QUADS, 0, self.num_vertices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_water(self):
        self.check_mesh_update()
        if self.is_dirty or self.num_water_vertices == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.water_vbo_id)
        self.set_water_material()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 8 * 4, ctypes.c_void_p(0))
        glTexCoordPointer(2, GL_FLOAT, 8 * 4, ctypes.c_void_p(3 * 4))
        glNormalPointer(GL_FLOAT, 8 * 4, ctypes.c_void_p(5 * 4))
        glDrawArrays(GL_QUADS, 0, self.num_water_vertices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_leaves(self):
        self.check_mesh_update()
        if self.is_dirty or self.num_leaves_vertices == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.leaves_vbo_id)
        self.set_leaves_material()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 8 * 4, ctypes.c_void_p(0))
        glTexCoordPointer(2, GL_FLOAT, 8 * 4, ctypes.c_void_p(3 * 4))
        glNormalPointer(GL_FLOAT, 8 * 4, ctypes.c_void_p(5 * 4))
        glDrawArrays(GL_QUADS, 0, self.num_leaves_vertices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

    def render_sand(self):
        self.check_mesh_update()
        if self.is_dirty or self.num_sand_vertices == 0:
            return
        glBindBuffer(GL_ARRAY_BUFFER, self.sand_vbo_id)
        self.set_sand_material()
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        glVertexPointer(3, GL_FLOAT, 8 * 4, ctypes.c_void_p(0))
        glTexCoordPointer(2, GL_FLOAT, 8 * 4, ctypes.c_void_p(3 * 4))
        glNormalPointer(GL_FLOAT, 8 * 4, ctypes.c_void_p(5 * 4))
        glDrawArrays(GL_QUADS, 0, self.num_sand_vertices)
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glBindBuffer(GL_ARRAY_BUFFER, 0)

# --- World Management ---
class World:
    def __init__(self):
        self.chunks = {}
        self.pending_chunks = set()
        self.max_preload_distance = VIEW_DISTANCE + 2

    def preload_chunks(self, player_x, player_z):
        chunk_x = int(player_x // CHUNK_SIZE)
        chunk_z = int(player_z // CHUNK_SIZE)
        for x in range(chunk_x - self.max_preload_distance, chunk_x + self.max_preload_distance + 1):
            for z in range(chunk_z - self.max_preload_distance, chunk_z + self.max_preload_distance + 1):
                if (x, z) not in self.chunks and (x, z) not in self.pending_chunks:
                    self.pending_chunks.add((x, z))
                    executor.submit(self.preload_chunk, x, z)

    def preload_chunk(self, x, z):
        chunk = Chunk(x, z)
        with threading.Lock():
            self.chunks[(x, z)] = chunk
        self.pending_chunks.remove((x, z))

    def get_chunk(self, x, z):
        if (x, z) not in self.chunks:
            self.chunks[(x, z)] = Chunk(x, z)
        return self.chunks[(x, z)]

    def render(self, player_x, player_z):
        chunk_x = int(player_x // CHUNK_SIZE)
        chunk_z = int(player_z // CHUNK_SIZE)
        proj_matrix = glGetDoublev(GL_PROJECTION_MATRIX)
        modelview_matrix = glGetDoublev(GL_MODELVIEW_MATRIX)
        planes = extract_frustum_planes(proj_matrix, modelview_matrix)

        # Render terrain
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                if is_chunk_in_frustum(x, z, planes):
                    chunk = self.get_chunk(x, z)
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render()
                    glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # Render sand
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                if is_chunk_in_frustum(x, z, planes):
                    chunk = self.get_chunk(x, z)
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_sand()
                    glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # Render leaves with blending
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                if is_chunk_in_frustum(x, z, planes):
                    chunk = self.get_chunk(x, z)
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_leaves()
                    glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)

        # Render water with blending
        glDepthMask(GL_FALSE)
        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_TEXTURE_COORD_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                if is_chunk_in_frustum(x, z, planes):
                    chunk = self.get_chunk(x, z)
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_water()
                    glPopMatrix()
        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_TEXTURE_COORD_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)

    def unload_far_chunks(self, player_chunk_x, player_chunk_z, max_distance):
        keys_to_delete = []
        for (cx, cz) in list(self.chunks.keys()):
            if abs(cx - player_chunk_x) > max_distance or abs(cz - player_chunk_z) > max_distance:
                keys_to_delete.append((cx, cz))
        for key in keys_to_delete:
            chunk = self.chunks.pop(key)
            glDeleteBuffers(1, [chunk.vbo_id])
            glDeleteBuffers(1, [chunk.water_vbo_id])
            glDeleteBuffers(1, [chunk.leaves_vbo_id])
            glDeleteBuffers(1, [chunk.sand_vbo_id])

# --- Player ---
class Player:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z
        self.yaw = 0
        self.pitch = 0
        self.on_ground = False

    def update(self, world):
        dx = 0
        dz = 0
        keys = pygame.key.get_pressed()
        if keys[K_w]:
            dx += math.sin(math.radians(self.yaw)) * MOVE_SPEED
            dz -= math.cos(math.radians(self.yaw)) * MOVE_SPEED
        if keys[K_s]:
            dx -= math.sin(math.radians(self.yaw)) * MOVE_SPEED
            dz += math.cos(math.radians(self.yaw)) * MOVE_SPEED
        if keys[K_a]:
            dx -= math.cos(math.radians(self.yaw)) * MOVE_SPEED
            dz -= math.sin(math.radians(self.yaw)) * MOVE_SPEED
        if keys[K_d]:
            dx += math.cos(math.radians(self.yaw)) * MOVE_SPEED
            dz += math.sin(math.radians(self.yaw)) * MOVE_SPEED
        if keys[K_SPACE]:
            self.y += MOVE_SPEED
        if keys[K_LSHIFT]:
            self.y -= MOVE_SPEED

        new_x = self.x + dx
        new_y = self.y
        new_z = self.z + dz

        ground_y = int(new_y - 0.1)
        if self.is_block_at(world, new_x, ground_y, new_z):
            new_y = ground_y + 1
            self.on_ground = True
        else:
            self.on_ground = False

        if self.is_block_at(world, new_x, new_y, new_z):
            new_x = self.x
        if self.is_block_at(world, new_x, new_y, new_z):
            new_z = self.z

        self.x = new_x
        self.y = new_y
        self.z = new_z

        mouse_x, mouse_y = pygame.mouse.get_rel()
        self.yaw = (self.yaw + mouse_x * ROTATION_SPEED) % 360
        self.pitch = max(-90, min(90, self.pitch + mouse_y * ROTATION_SPEED))

    def is_block_at(self, world, x, y, z):
        chunk_x = int(x // CHUNK_SIZE)
        chunk_z = int(z // CHUNK_SIZE)
        block_x = int(x % CHUNK_SIZE)
        block_z = int(z % CHUNK_SIZE)
        try:
            chunk = world.get_chunk(chunk_x, chunk_z)
            block_type = chunk.blocks.get((block_x, int(y), block_z))
            return block_type is not None and block_type not in ["water", "leaves"]
        except KeyError:
            return False

# --- Texture Loading ---
def load_texture(filename):
    texture_surface = pygame.image.load(filename).convert_alpha()
    texture_data = pygame.image.tostring(texture_surface, "RGBA", 1)
    width = texture_surface.get_width()
    height = texture_surface.get_height()
    texture_id = glGenTextures(1)
    glBindTexture(GL_TEXTURE_2D, texture_id)
    glTexImage2D(GL_TEXTURE_2D, 0, GL_RGBA, width, height, 0,
                 GL_RGBA, GL_UNSIGNED_BYTE, texture_data)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_LINEAR)
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_LINEAR)
    if "water" in filename or "leaves" in filename or "sand" in filename:
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_REPEAT)
        glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_REPEAT)
    return texture_id

# --- UI Helper Functions ---
def draw_text(x, y, text, font, color=(255, 255, 255)):
    text_surface = font.render(text, True, color)
    text_data = pygame.image.tostring(text_surface, "RGBA", True)
    glRasterPos2f(x, y)
    glDrawPixels(text_surface.get_width(), text_surface.get_height(), GL_RGBA, GL_UNSIGNED_BYTE, text_data)

def draw_menu(fov):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    font = pygame.font.SysFont("Arial", 24)
    draw_text(50, 50, "Main Menu", font)
    draw_text(50, 80, "Settings", font)
    draw_text(50, 110, f"FOV: {fov}", font)
    draw_text(50, 140, "Press UP to increase, DOWN to decrease FOV", font)
    draw_text(50, 170, "Press M to resume", font)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

def draw_fps(fps):
    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    glOrtho(0, SCREEN_WIDTH, SCREEN_HEIGHT, 0, -1, 1)
    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    font = pygame.font.SysFont("Arial", 24)
    draw_text(10, 10, f"FPS: {fps:.2f}", font)

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glPopMatrix()
    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)

# --- Main Function ---
def main():
    pygame.init()
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Procedural World")
    pygame.mouse.set_visible(False)
    pygame.event.set_grab(True)

    terrain_texture_id = load_texture("Rohan_Python_Sims/Images/terrainsmall.jpg")
    water_texture_id = load_texture("Rohan_Python_Sims/Images/water.jpg")
    leaves_texture_id = load_texture("Rohan_Python_Sims/Images/leaves.jpg")
    sand_texture_id = load_texture("Rohan_Python_Sims/Images/sand.jpg")  # Load sand texture

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_TEXTURE_2D)
    setup_lighting()

    world = World()
    player = Player(0, get_height(0, 0) + 2, 0)
    clock = pygame.time.Clock()

    fov = 80
    show_menu = False
    show_fps = False

    while True:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                quit()
            if event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    pygame.quit()
                    quit()
                elif event.key == pygame.K_m:
                    show_menu = not show_menu
                elif event.key == pygame.K_f:
                    show_fps = not show_fps
                elif show_menu:
                    if event.key == pygame.K_UP:
                        fov = min(fov + 1, 120)
                    elif event.key == pygame.K_DOWN:
                        fov = max(fov - 1, 30)

        if not show_menu:
            player.update(world)
            world.preload_chunks(player.x, player.z)

        glClearColor(0.6, 0.8, 1.0, 1.0)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)

        glMatrixMode(GL_PROJECTION)
        glLoadIdentity()
        gluPerspective(fov, (SCREEN_WIDTH / SCREEN_HEIGHT), 0.1, 500.0)

        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        glRotatef(player.pitch, 1, 0, 0)
        glRotatef(player.yaw, 0, 1, 0)
        glTranslatef(-player.x, -player.y, -player.z)

        light_pos = (player.x + 10, player.y + 20, player.z + 10, 0)
        glLightfv(GL_LIGHT0, GL_POSITION, light_pos)

        # Render terrain
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(1.0, 1.0)
        glBindTexture(GL_TEXTURE_2D, terrain_texture_id)
        chunk_x = int(player.x // CHUNK_SIZE)
        chunk_z = int(player.z // CHUNK_SIZE)
        world.unload_far_chunks(chunk_x, chunk_z, VIEW_DISTANCE + 2)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                chunk = world.get_chunk(x, z)
                if chunk:
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render()
                    glPopMatrix()
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Render sand
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(0.5, 0.5)  # Slight offset to avoid z-fighting with terrain
        glBindTexture(GL_TEXTURE_2D, sand_texture_id)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                chunk = world.get_chunk(x, z)
                if chunk:
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_sand()
                    glPopMatrix()
        glDisable(GL_POLYGON_OFFSET_FILL)

        # Render leaves
        glEnable(GL_POLYGON_OFFSET_FILL)
        glPolygonOffset(-1.0, -1.0)
        glEnable(GL_BLEND)
        glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
        glBindTexture(GL_TEXTURE_2D, leaves_texture_id)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                chunk = world.get_chunk(x, z)
                if chunk:
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_leaves()
                    glPopMatrix()

        # Render water
        glBindTexture(GL_TEXTURE_2D, water_texture_id)
        for x in range(chunk_x - VIEW_DISTANCE, chunk_x + VIEW_DISTANCE + 1):
            for z in range(chunk_z - VIEW_DISTANCE, chunk_z + VIEW_DISTANCE + 1):
                chunk = world.get_chunk(x, z)
                if chunk:
                    glPushMatrix()
                    glTranslatef(x * CHUNK_SIZE, 0, z * CHUNK_SIZE)
                    chunk.render_water()
                    glPopMatrix()
        glDepthMask(GL_TRUE)
        glDisable(GL_BLEND)
        glDisable(GL_POLYGON_OFFSET_FILL)

        if show_menu:
            draw_menu(fov)
        if show_fps:
            draw_fps(clock.get_fps())

        pygame.display.flip()
        clock.tick(60)

if __name__ == "__main__":
    main()