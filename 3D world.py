import pygame
from pygame.locals import DOUBLEBUF, OPENGL, QUIT, MOUSEBUTTONDOWN, MOUSEBUTTONUP, MOUSEMOTION, KEYDOWN, K_SPACE, K_UP, K_DOWN
from pygame.locals import K_w, K_a, K_s, K_d, K_q, K_e, K_r, K_f, K_c, K_v, K_1, K_2, K_3, K_ESCAPE, K_LSHIFT
from OpenGL.GL import *
from OpenGL.GLU import *
import math
import random
import time

# Global variables
SCREEN_WIDTH, SCREEN_HEIGHT = 1024, 768
DEBUG_MODE = False

class Aircraft:
    def __init__(self):
        self.position = [0, 3, 0]  # [x, y, z]
        self.velocity = [0, 0, 0]  # [vx, vy, vz]
        self.acceleration = [0, 0, 0]  # [ax, ay, az]
        self.orientation = [0, 0, 0]  # [pitch, yaw, roll] in degrees
        self.angular_velocity = [0, 0, 0]  # [pitch_rate, yaw_rate, roll_rate] in degrees/sec
        
        # Flight characteristics
        self.max_speed = 0.5
        self.thrust = 0.01
        self.lift_coefficient = 0.005
        self.drag_coefficient = 0.001
        self.turn_rate = 1.0
        self.pitch_rate = 0.8
        self.roll_rate = 1.5
        
        # Control inputs
        self.throttle = 0.0  # 0.0 to 1.0
        self.pitch_input = 0.0  # -1.0 to 1.0
        self.yaw_input = 0.0  # -1.0 to 1.0
        self.roll_input = 0.0  # -1.0 to 1.0
        
        # Camera
        self.camera_mode = 0  # 0: cockpit, 1: chase, 2: external
        self.camera_distance = 5.0
        self.camera_elevation = 1.0
        
    def update(self, dt):
        # Apply throttle
        forward_vector = self.get_forward_vector()
        self.acceleration[0] = forward_vector[0] * self.throttle * self.thrust
        self.acceleration[1] = forward_vector[1] * self.throttle * self.thrust
        self.acceleration[2] = forward_vector[2] * self.throttle * self.thrust
        
        # Apply gravity
        self.acceleration[1] -= 0.01
        
        # Apply lift (simplified)
        speed = math.sqrt(sum(v*v for v in self.velocity))
        if speed > 0.1:
            up_vector = self.get_up_vector()
            lift = speed * self.lift_coefficient
            self.acceleration[0] += up_vector[0] * lift
            self.acceleration[1] += up_vector[1] * lift
            self.acceleration[2] += up_vector[2] * lift
        
        # Apply drag
        if speed > 0:
            drag = speed * speed * self.drag_coefficient
            self.acceleration[0] -= self.velocity[0] * drag / speed
            self.acceleration[1] -= self.velocity[1] * drag / speed
            self.acceleration[2] -= self.velocity[2] * drag / speed
        
        # Update velocity
        for i in range(3):
            self.velocity[i] += self.acceleration[i] * dt
        
        # Limit speed
        speed = math.sqrt(sum(v*v for v in self.velocity))
        if speed > self.max_speed:
            scale = self.max_speed / speed
            for i in range(3):
                self.velocity[i] *= scale
        
        # Update position
        for i in range(3):
            self.position[i] += self.velocity[i] * dt
        
        # Prevent crashing into the ground (improved collision)
        if self.position[1] < 0.1:
            self.position[1] = 0.1
            self.velocity[1] = max(0, self.velocity[1])  # Stop downward movement
            self.acceleration[1] = 0  # Reset vertical acceleration
        
        # Update orientation based on control inputs
        self.orientation[0] += self.pitch_input * self.pitch_rate * dt * 60
        self.orientation[1] += self.yaw_input * self.turn_rate * dt * 60
        self.orientation[2] += self.roll_input * self.roll_rate * dt * 60
        
        # Normalize angles
        self.orientation[0] = (self.orientation[0] + 180) % 360 - 180  # Pitch: -180 to 180
        self.orientation[1] = self.orientation[1] % 360  # Yaw: 0 to 360
        self.orientation[2] = (self.orientation[2] + 180) % 360 - 180  # Roll: -180 to 180
        
        # Reset acceleration
        self.acceleration = [0, 0, 0]
    
    def get_forward_vector(self):
        # Convert orientation to radians
        pitch_rad = math.radians(self.orientation[0])
        yaw_rad = math.radians(self.orientation[1])
        
        # Calculate forward direction
        x = -math.sin(yaw_rad) * math.cos(pitch_rad)
        y = math.sin(pitch_rad)
        z = -math.cos(yaw_rad) * math.cos(pitch_rad)
        
        return [x, y, z]
    
    def get_up_vector(self):
        # Convert orientation to radians
        pitch_rad = math.radians(self.orientation[0])
        yaw_rad = math.radians(self.orientation[1])
        roll_rad = math.radians(self.orientation[2])
        
        # Calculate up direction
        x = math.sin(roll_rad) * math.sin(yaw_rad) - math.cos(roll_rad) * math.sin(pitch_rad) * math.cos(yaw_rad)
        y = math.cos(roll_rad) * math.cos(pitch_rad)
        z = -math.sin(roll_rad) * math.cos(yaw_rad) - math.cos(roll_rad) * math.sin(pitch_rad) * math.sin(yaw_rad)
        
        return [x, y, z]
    
    def get_right_vector(self):
        # Cross product of forward and up vectors
        forward = self.get_forward_vector()
        up = self.get_up_vector()
        
        right = [
            up[1] * forward[2] - up[2] * forward[1],
            up[2] * forward[0] - up[0] * forward[2],
            up[0] * forward[1] - up[1] * forward[0]
        ]
        
        # Normalize
        magnitude = math.sqrt(sum(v*v for v in right))
        if magnitude > 0:
            right = [v / magnitude for v in right]
        
        return right
    
    def setup_camera(self):
        # Set up the camera based on the aircraft position and orientation
        glMatrixMode(GL_MODELVIEW)
        glLoadIdentity()
        
        if self.camera_mode == 0:  # Cockpit view
            # Calculate the camera position and target
            forward = self.get_forward_vector()
            up = self.get_up_vector()
            
            # Look from slightly above the aircraft
            eye_offset = [0, 0.3, 0]  # Offset from the aircraft position
            target_offset = [forward[0], forward[1], forward[2]]  # Look forward
            
            gluLookAt(
                self.position[0] + eye_offset[0], self.position[1] + eye_offset[1], self.position[2] + eye_offset[2],
                self.position[0] + target_offset[0], self.position[1] + target_offset[1], self.position[2] + target_offset[2],
                up[0], up[1], up[2]
            )
        
        elif self.camera_mode == 1:  # Chase view
            # Calculate camera position behind the aircraft
            forward = self.get_forward_vector()
            up = self.get_up_vector()
            
            # Position the camera behind and above the aircraft
            camera_pos = [
                self.position[0] - forward[0] * self.camera_distance,
                self.position[1] - forward[1] * self.camera_distance + self.camera_elevation,
                self.position[2] - forward[2] * self.camera_distance
            ]
            
            gluLookAt(
                camera_pos[0], camera_pos[1], camera_pos[2],
                self.position[0], self.position[1], self.position[2],
                up[0], up[1], up[2]
            )
        
        else:  # External view
            # Orbit around the aircraft
            time_offset = pygame.time.get_ticks() / 5000.0
            orbit_radius = self.camera_distance * 2
            orbit_height = self.camera_elevation * 2
            
            camera_pos = [
                self.position[0] + orbit_radius * math.sin(time_offset),
                self.position[1] + orbit_height,
                self.position[2] + orbit_radius * math.cos(time_offset)
            ]
            
            gluLookAt(
                camera_pos[0], camera_pos[1], camera_pos[2],
                self.position[0], self.position[1], self.position[2],
                0, 1, 0
            )
    
    def draw(self):
        # Save the current matrix
        glPushMatrix()
        
        # Position the aircraft
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Rotate in order: yaw, pitch, roll
        glRotatef(self.orientation[1], 0, 1, 0)  # Yaw
        glRotatef(self.orientation[0], 1, 0, 0)  # Pitch
        glRotatef(self.orientation[2], 0, 0, 1)  # Roll
        
        # Draw the aircraft body
        glColor3f(0.7, 0.7, 0.7)
        
        # Fuselage
        glPushMatrix()
        glScalef(0.15, 0.1, 0.5)
        draw_cube()
        glPopMatrix()
        
        # Wings
        glPushMatrix()
        glTranslatef(0, 0, 0)
        glScalef(0.6, 0.02, 0.2)
        draw_cube()
        glPopMatrix()
        
        # Tail
        glPushMatrix()
        glTranslatef(0, 0.1, -0.4)
        glScalef(0.03, 0.15, 0.1)
        draw_cube()
        glPopMatrix()
        
        # Horizontal stabilizer
        glPushMatrix()
        glTranslatef(0, 0, -0.4)
        glScalef(0.3, 0.02, 0.1)
        draw_cube()
        glPopMatrix()
        
        # Restore the matrix
        glPopMatrix()
        
        # Draw contrails
        self.draw_contrails()
    
    def draw_contrails(self):
        # Draw contrails based on speed
        speed = math.sqrt(sum(v*v for v in self.velocity))
        if speed > 0.2:
            glPushMatrix()
            glDisable(GL_LIGHTING)
            glEnable(GL_BLEND)
            glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
            
            glBegin(GL_LINES)
            glColor4f(1, 1, 1, 0.5)
            
            # Left wing contrail
            glVertex3f(self.position[0] - 0.3, self.position[1], self.position[2])
            backward = [-v for v in self.get_forward_vector()]
            glVertex3f(
                self.position[0] - 0.3 + backward[0] * speed * 2,
                self.position[1] + backward[1] * speed * 2,
                self.position[2] + backward[2] * speed * 2
            )
            
            # Right wing contrail
            glVertex3f(self.position[0] + 0.3, self.position[1], self.position[2])
            glVertex3f(
                self.position[0] + 0.3 + backward[0] * speed * 2,
                self.position[1] + backward[1] * speed * 2,
                self.position[2] + backward[2] * speed * 2
            )
            
            glEnd()
            
            glDisable(GL_BLEND)
            glEnable(GL_LIGHTING)
            glPopMatrix()

class Cloud:
    def __init__(self, position, size):
        self.position = position
        self.size = size
        self.texture_offset = random.uniform(0, 1)
    
    def draw(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Draw cloud as a semi-transparent billboarded quad
        modelview = glGetDoublev(GL_MODELVIEW_MATRIX)
        right = [modelview[0,0], modelview[1,0], modelview[2,0]]  # First column
        up = [modelview[0,1], modelview[1,1], modelview[2,1]]     # Second column
        
        half_size = self.size / 2
        glBegin(GL_QUADS)
        glTexCoord2f(0, 0)
        glVertex3f(-half_size * right[0] - half_size * up[0], 
                   -half_size * right[1] - half_size * up[1], 
                   -half_size * right[2] - half_size * up[2])
        glTexCoord2f(1, 0)
        glVertex3f(half_size * right[0] - half_size * up[0], 
                   half_size * right[1] - half_size * up[1], 
                   half_size * right[2] - half_size * up[2])
        glTexCoord2f(1, 1)
        glVertex3f(half_size * right[0] + half_size * up[0], 
                   half_size * right[1] + half_size * up[1], 
                   half_size * right[2] + half_size * up[2])
        glTexCoord2f(0, 1)
        glVertex3f(-half_size * right[0] + half_size * up[0], 
                   -half_size * right[1] + half_size * up[1], 
                   -half_size * right[2] + half_size * up[2])
        glEnd()
        
        glPopMatrix()

class Building:
    def __init__(self, position, width, height, depth):
        self.position = position
        self.width = width
        self.height = height
        self.depth = depth
        self.color = (
            random.uniform(0.4, 0.8),
            random.uniform(0.4, 0.8),
            random.uniform(0.4, 0.8)
        )
        self.has_windows = random.choice([True, False])
        
    def draw(self):
        glPushMatrix()
        glTranslatef(self.position[0], self.position[1], self.position[2])
        
        # Main building body
        glColor3f(*self.color)
        glPushMatrix()
        glScalef(self.width, self.height, self.depth)
        draw_cube()
        glPopMatrix()
        
        # Draw windows if applicable
        if self.has_windows:
            glColor3f(0.9, 0.9, 0.4)  # Yellow window light
            window_size = min(self.width, self.depth) * 0.15
            window_depth = 0.01
            
            # Front windows
            for x in range(int(self.width / window_size) - 1):
                for y in range(int(self.height / window_size) - 1):
                    if random.random() < 0.7:  # Some windows are dark
                        glPushMatrix()
                        glTranslatef(
                            (x + 1) * window_size - self.width/2 + window_size/2,
                            (y + 1) * window_size - self.height/2 + window_size*2,
                            self.depth/2 + window_depth/2
                        )
                        glScalef(window_size * 0.7, window_size * 0.7, window_depth)
                        draw_cube()
                        glPopMatrix()
            
            # Back windows
            for x in range(int(self.width / window_size) - 1):
                for y in range(int(self.height / window_size) - 1):
                    if random.random() < 0.7:  # Some windows are dark
                        glPushMatrix()
                        glTranslatef(
                            (x + 1) * window_size - self.width/2 + window_size/2,
                            (y + 1) * window_size - self.height/2 + window_size*2,
                            -self.depth/2 - window_depth/2
                        )
                        glScalef(window_size * 0.7, window_size * 0.7, window_depth)
                        draw_cube()
                        glPopMatrix()
        
        glPopMatrix()

def generate_earth_texture():
    """Generate a 512x256 surface with a procedural 'Earth-like' texture."""
    size_x, size_y = 512, 256
    surface = pygame.Surface((size_x, size_y), pygame.SRCALPHA)
    surface.fill((0, 64, 128))  # deep blue
    
    # Create a grid for the land/water
    grid_size = 64
    noise_scale = 0.2
    
    # Generate continents
    for x in range(size_x):
        for y in range(size_y):
            # Create Perlin-like noise
            nx = x / size_x - 0.5
            ny = y / size_y - 0.5
            
            # Simple noise function
            noise = math.sin(nx * 5) * math.cos(ny * 3) + math.sin(nx * 13) * math.cos(ny * 7)
            
            # Add latitude-based coloring (colder at poles)
            latitude_factor = abs(y - size_y/2) / (size_y/2)
            
            if noise > 0:
                # Land
                green = max(0, min(255, 100 - latitude_factor * 60))
                brown = max(0, min(255, 70 + latitude_factor * 50))
                surface.set_at((x, y), (brown, green, 0))
                
                # Add snow at the poles
                if latitude_factor > 0.7:
                    snow_amount = (latitude_factor - 0.7) / 0.3
                    white_amount = int(snow_amount * 255)
                    current = surface.get_at((x, y))
                    surface.set_at((x, y), (
                        int(current[0] * (1 - snow_amount) + white_amount),
                        int(current[1] * (1 - snow_amount) + white_amount),
                        int(current[2] * (1 - snow_amount) + white_amount)
                    ))
            else:
                # Water
                blue = max(0, min(255, 150 - latitude_factor * 50))
                surface.set_at((x, y), (0, 64, blue))
    
    # Add some coastal details
    for x in range(1, size_x-1):
        for y in range(1, size_y-1):
            current = surface.get_at((x, y))
            if current[2] > 100:  # If water
                # Check if there's land nearby
                has_land = False
                for dx in [-1, 0, 1]:
                    for dy in [-1, 0, 1]:
                        if dx == 0 and dy == 0:
                            continue
                        neighbor = surface.get_at((x+dx, y+dy))
                        if neighbor[2] < 50:  # Land is detected
                            has_land = True
                
                # If it's a coastal cell, make it lighter blue
                if has_land:
                    surface.set_at((x, y), (0, 100, 180))
    
    return surface

def generate_cloud_texture():
    """Generate a cloud texture."""
    size = 256
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    surface.fill((0, 0, 0, 0))  # Transparent
    
    # Draw cloud formations
    for _ in range(15):
        x = random.randint(0, size)
        y = random.randint(0, size)
        radius = random.randint(20, 60)
        
        # Draw a soft, circular cloud puff
        for dx in range(-radius, radius+1):
            for dy in range(-radius, radius+1):
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > radius:
                    continue
                
                px, py = x + dx, y + dy
                if 0 <= px < size and 0 <= py < size:
                    alpha = int(255 * (1 - dist/radius))
                    current = surface.get_at((px, py))
                    new_alpha = min(255, current[3] + alpha)
                    surface.set_at((px, py), (255, 255, 255, new_alpha))
    
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

def draw_cube():
    """Draw a unit cube centered at the origin."""
    vertices = [
        # Front face
        (-0.5, -0.5, 0.5), (0.5, -0.5, 0.5), (0.5, 0.5, 0.5), (-0.5, 0.5, 0.5),
        # Back face
        (-0.5, -0.5, -0.5), (-0.5, 0.5, -0.5), (0.5, 0.5, -0.5), (0.5, -0.5, -0.5),
        # Top face
        (-0.5, 0.5, -0.5), (-0.5, 0.5, 0.5), (0.5, 0.5, 0.5), (0.5, 0.5, -0.5),
        # Bottom face
        (-0.5, -0.5, -0.5), (0.5, -0.5, -0.5), (0.5, -0.5, 0.5), (-0.5, -0.5, 0.5),
        # Right face
        (0.5, -0.5, -0.5), (0.5, 0.5, -0.5), (0.5, 0.5, 0.5), (0.5, -0.5, 0.5),
        # Left face
        (-0.5, -0.5, -0.5), (-0.5, -0.5, 0.5), (-0.5, 0.5, 0.5), (-0.5, 0.5, -0.5)
    ]
    
    normals = [
        (0, 0, 1), (0, 0, 1), (0, 0, 1), (0, 0, 1),  # Front
        (0, 0, -1), (0, 0, -1), (0, 0, -1), (0, 0, -1),  # Back
        (0, 1, 0), (0, 1, 0), (0, 1, 0), (0, 1, 0),  # Top
        (0, -1, 0), (0, -1, 0), (0, -1, 0), (0, -1, 0),  # Bottom
        (1, 0, 0), (1, 0, 0), (1, 0, 0), (1, 0, 0),  # Right
        (-1, 0, 0), (-1, 0, 0), (-1, 0, 0), (-1, 0, 0)  # Left
    ]
    
    texcoords = [
        (0, 0), (1, 0), (1, 1), (0, 1),  # Front
        (1, 0), (1, 1), (0, 1), (0, 0),  # Back
        (0, 1), (0, 0), (1, 0), (1, 1),  # Top
        (1, 1), (0, 1), (0, 0), (1, 0),  # Bottom
        (1, 0), (1, 1), (0, 1), (0, 0),  # Right
        (0, 0), (1, 0), (1, 1), (0, 1)   # Left
    ]
    
    indices = [
        0, 1, 2, 2, 3, 0,  # Front
        4, 5, 6, 6, 7, 4,  # Back
        8, 9, 10, 10, 11, 8,  # Top
        12, 13, 14, 14, 15, 12,  # Bottom
        16, 17, 18, 18, 19, 16,  # Right
        20, 21, 22, 22, 23, 20   # Left
    ]
    
    glBegin(GL_TRIANGLES)
    for i in indices:
        glNormal3fv(normals[i])
        glTexCoord2fv(texcoords[i])
        glVertex3fv(vertices[i])
    glEnd()

def draw_earth(texture_id, position=[0, 0, 0], scale=1.0):
    """Draw the Earth as a textured sphere."""
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    glRotatef(90, 1, 0, 0)  # Orient poles along Y axis
    
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    # Draw Earth sphere
    glColor3f(1, 1, 1)
    quadric = gluNewQuadric()
    gluQuadricTexture(quadric, GL_TRUE)
    gluSphere(quadric, scale, 50, 50)
    gluDeleteQuadric(quadric)
    
    glPopMatrix()

def draw_sky(radius=100):
    """Draw a simple sky dome."""
    glPushMatrix()
    glDisable(GL_LIGHTING)
    
    # Sky gradient from bottom to top
    glBegin(GL_TRIANGLE_FAN)
    glColor3f(0.4, 0.7, 1.0)  # Zenith color (light blue)
    glVertex3f(0, radius, 0)  # Top vertex
    
    # Circle of vertices at the horizon
    steps = 32
    for i in range(steps + 1):
        angle = 2 * math.pi * i / steps
        x = radius * math.cos(angle)
        z = radius * math.sin(angle)
        glColor3f(0.7, 0.8, 0.9)  # Horizon color
        glVertex3f(x, 0, z)
    
    glEnd()
    
    glEnable(GL_LIGHTING)
    glPopMatrix()

def draw_runway(length=15, width=1.5, position=[0, 0, 0]):
    """Draw a simple runway."""
    glPushMatrix()
    glTranslatef(position[0], position[1], position[2])
    
    # Main runway surface
    glColor3f(0.3, 0.3, 0.3)  # Dark gray
    glBegin(GL_QUADS)
    glVertex3f(-width/2, 0.01, -length/2)
    glVertex3f(width/2, 0.01, -length/2)
    glVertex3f(width/2, 0.01, length/2)
    glVertex3f(-width/2, 0.01, length/2)
    glEnd()
    
    # Runway center markings
    glColor3f(1, 1, 1)  # White
    center_width = 0.2
    for i in range(10):
        marker_length = 0.8
        z_pos = -length/2 + length * i / 10 + length/20
        
        glBegin(GL_QUADS)
        glVertex3f(-center_width/2, 0.02, z_pos)
        glVertex3f(center_width/2, 0.02, z_pos)
        glVertex3f(center_width/2, 0.02, z_pos + marker_length)
        glVertex3f(-center_width/2, 0.02, z_pos)
        glVertex3f(center_width/2, 0.02, z_pos)
        glVertex3f(center_width/2, 0.02, z_pos + marker_length)
        glVertex3f(-center_width/2, 0.02, z_pos + marker_length)
        glEnd()
    
    # Runway edge markings
    for i in range(2):
        edge_pos = width/2 - 0.1 if i == 0 else -width/2 + 0.1
        
        glBegin(GL_QUADS)
        glVertex3f(edge_pos - 0.05, 0.02, -length/2)
        glVertex3f(edge_pos + 0.05, 0.02, -length/2)
        glVertex3f(edge_pos + 0.05, 0.02, length/2)
        glVertex3f(edge_pos - 0.05, 0.02, length/2)
        glEnd()
    
    glPopMatrix()

def draw_hud(screen, aircraft, fps):
    """Draw a heads-up display with flight information."""
    font = pygame.font.Font(None, 24)
    
    # Calculate actual speed in km/h for display
    speed_kmh = math.sqrt(sum(v*v for v in aircraft.velocity)) * 500
    altitude = aircraft.position[1] * 1000  # Scale for display
    
    # FPS counter
    fps_text = font.render(f"FPS: {fps:.1f}", True, (255, 255, 255))
    screen.blit(fps_text, (10, 10))
    
    # Flight information
    info_texts = [
        f"Speed: {speed_kmh:.1f} km/h",
        f"Altitude: {altitude:.1f} m",
        f"Pitch: {aircraft.orientation[0]:.1f}°",
        f"Heading: {aircraft.orientation[1]:.1f}°",
        f"Roll: {aircraft.orientation[2]:.1f}°",
        f"Throttle: {aircraft.throttle*100:.0f}%",
        f"Camera: {'Cockpit' if aircraft.camera_mode == 0 else 'Chase' if aircraft.camera_mode == 1 else 'External'}"
    ]
    
    y_offset = 40
    for text in info_texts:
        rendered_text = font.render(text, True, (255, 255, 255))
        screen.blit(rendered_text, (10, y_offset))
        y_offset += 25
    
    # Draw artificial horizon in cockpit view
    if aircraft.camera_mode == 0:
        center_x, center_y = screen.get_width() // 2, screen.get_height() // 2
        horizon_size = 150
        
        # Horizon line adjusted for pitch and roll
        pitch_offset = aircraft.orientation[0] * 2  # 2 pixels per degree
        roll_rad = math.radians(aircraft.orientation[2])
        
        # Draw background (sky/ground)
        pygame.draw.circle(screen, (0, 0, 0), (center_x, center_y), horizon_size + 5)
        
        # Calculate horizon line endpoints
        dx = math.cos(roll_rad + math.pi/2) * horizon_size
        dy = math.sin(roll_rad + math.pi/2) * horizon_size
        
        # Calculate horizon line offset for pitch
        pitch_dx = math.sin(roll_rad) * pitch_offset
        pitch_dy = -math.cos(roll_rad) * pitch_offset
        
        # Draw horizon line
        pygame.draw.line(
            screen, 
            (255, 255, 255),
            (center_x - dx + pitch_dx, center_y - dy + pitch_dy),
            (center_x + dx + pitch_dx, center_y + dy + pitch_dy),
            2
        )
        
        # Draw pitch ladder
        for pitch in [-20, -10, 10, 20]:
            pitch_line_offset = pitch * 2  # 2 pixels per degree
            pitch_total_dy = -math.cos(roll_rad) * pitch_line_offset
            pitch_total_dx = math.sin(roll_rad) * pitch_line_offset
            
            # Draw shorter lines for pitch marks
            line_length = 20
            pygame.draw.line(
                screen, 
                (255, 255, 255),
                (center_x - line_length*math.cos(roll_rad) + pitch_dx + pitch_total_dx, 
                 center_y - line_length*math.sin(roll_rad) + pitch_dy + pitch_total_dy),
                (center_x + line_length*math.cos(roll_rad) + pitch_dx + pitch_total_dx, 
                 center_y + line_length*math.sin(roll_rad) + pitch_dy + pitch_total_dy),
                1
            )
            
            # Draw pitch value
            pitch_text = font.render(f"{pitch}°", True, (255, 255, 255))
            text_x = center_x + pitch_dx + pitch_total_dx + 25*math.cos(roll_rad)
            text_y = center_y + pitch_dy + pitch_total_dy + 25*math.sin(roll_rad)
            screen.blit(pitch_text, (text_x, text_y))
        
        # Draw center crosshair
        crosshair_size = 10
        pygame.draw.line(screen, (0, 255, 0), 
                        (center_x - crosshair_size, center_y), 
                        (center_x + crosshair_size, center_y), 2)
        pygame.draw.line(screen, (0, 255, 0), 
                        (center_x, center_y - crosshair_size), 
                        (center_x, center_y + crosshair_size), 2)

def init_opengl(width, height):
    """Initialize OpenGL settings."""
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(60, width / float(height), 0.1, 100.0)
    glMatrixMode(GL_MODELVIEW)
    glLoadIdentity()
    
    # Enable depth testing and back-face culling
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_CULL_FACE)
    glCullFace(GL_BACK)
    
    # Enable texture mapping
    glEnable(GL_TEXTURE_2D)
    
    # Set up lighting
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)
    glEnable(GL_COLOR_MATERIAL)
    glColorMaterial(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE)
    
    # Ambient light (global illumination)
    glLightModelfv(GL_LIGHT_MODEL_AMBIENT, [0.2, 0.2, 0.2, 1.0])
    
    # Sun light (directional)
    glLightfv(GL_LIGHT0, GL_AMBIENT, [0.1, 0.1, 0.1, 1.0])
    glLightfv(GL_LIGHT0, GL_DIFFUSE, [1.0, 1.0, 0.9, 1.0])
    glLightfv(GL_LIGHT0, GL_POSITION, [0.5, 1.0, 0.5, 0.0])  # Directional light

def generate_terrain(size=20, resolution=20):
    """Generate a simple terrain grid."""
    vertices = []
    for z in range(resolution):
        for x in range(resolution):
            # Calculate actual x, z positions
            pos_x = (x / (resolution-1) - 0.5) * size
            pos_z = (z / (resolution-1) - 0.5) * size
            
            # Compute a simple height
            height = 0
            
            # Add some hills
            for hill in range(5):
                hill_x = random.uniform(-size/2, size/2)
                hill_z = random.uniform(-size/2, size/2)
                hill_height = random.uniform(0.2, 0.6)
                hill_width = random.uniform(2, 5)
                
                dist = math.sqrt((pos_x - hill_x)**2 + (pos_z - hill_z)**2)
                height += hill_height * math.exp(-(dist**2) / (2 * hill_width**2))
            
            vertices.append((pos_x, height, pos_z))
    
    return vertices, resolution

def draw_terrain(vertices, resolution, texture_id):
    """Draw the terrain mesh."""
    glBindTexture(GL_TEXTURE_2D, texture_id)
    
    glColor3f(1, 1, 1)
    
    # Draw terrain as a triangle mesh
    for z in range(resolution - 1):
        for x in range(resolution - 1):
            # Get the four corners of the current grid cell
            idx00 = z * resolution + x
            idx10 = z * resolution + (x + 1)
            idx01 = (z + 1) * resolution + x
            idx11 = (z + 1) * resolution + (x + 1)
            
            # Calculate vertex normals (simplified)
            v00, v10, v01, v11 = vertices[idx00], vertices[idx10], vertices[idx01], vertices[idx11]
            
            # Calculate texture coordinates
            tex_x1, tex_z1 = x / (resolution-1), z / (resolution-1)
            tex_x2, tex_z2 = (x+1) / (resolution-1), (z+1) / (resolution-1)
            
            # Draw the two triangles for this grid cell
            glBegin(GL_TRIANGLES)
            
            # Triangle 1: v00, v10, v11
            # Calculate normal
            u = (v10[0] - v00[0], v10[1] - v00[1], v10[2] - v00[2])
            v = (v11[0] - v10[0], v11[1] - v10[1], v11[2] - v10[2])
            normal = (
                u[1]*v[2] - u[2]*v[1],
                u[2]*v[0] - u[0]*v[2],
                u[0]*v[1] - u[1]*v[0]
            )
            mag = math.sqrt(sum(n*n for n in normal))
            if mag > 0:
                normal = tuple(n/mag for n in normal)
            
            glNormal3fv(normal)
            glTexCoord2f(tex_x1, tex_z1)
            glVertex3fv(v00)
            
            glTexCoord2f(tex_x2, tex_z1)
            glVertex3fv(v10)
            
            glTexCoord2f(tex_x2, tex_z2)
            glVertex3fv(v11)
            
            # Triangle 2: v00, v11, v01
            # Calculate normal
            u = (v11[0] - v00[0], v11[1] - v00[1], v11[2] - v00[2])
            v = (v01[0] - v00[0], v01[1] - v00[1], v01[2] - v00[2])
            normal = (
                u[1]*v[2] - u[2]*v[1],
                u[2]*v[0] - u[0]*v[2],
                u[0]*v[1] - u[1]*v[0]
            )
            mag = math.sqrt(sum(n*n for n in normal))
            if mag > 0:
                normal = tuple(n/mag for n in normal)
            
            glNormal3fv(normal)
            glTexCoord2f(tex_x1, tex_z1)
            glVertex3fv(v00)
            
            glTexCoord2f(tex_x2, tex_z2)
            glVertex3fv(v11)
            
            glTexCoord2f(tex_x1, tex_z2)
            glVertex3fv(v01)
            
            glEnd()

def generate_terrain_texture():
    """Generate a simple terrain texture with grass and dirt."""
    size = 256
    surface = pygame.Surface((size, size), pygame.SRCALPHA)
    
    # Base color (grass green)
    surface.fill((76, 153, 0))
    
    # Add some noise/variation
    for x in range(size):
        for y in range(size):
            # Add some noise
            noise = random.randint(-15, 15)
            
            # Get current color and adjust
            color = surface.get_at((x, y))
            new_color = (
                max(0, min(255, color[0] + noise)),
                max(0, min(255, color[1] + noise)),
                max(0, min(255, color[2] + noise)),
                255
            )
            surface.set_at((x, y), new_color)
    
    # Add some dirt patches
    for _ in range(20):
        patch_x = random.randint(0, size)
        patch_y = random.randint(0, size)
        patch_size = random.randint(5, 20)
        
        for dx in range(-patch_size, patch_size+1):
            for dy in range(-patch_size, patch_size+1):
                dist = math.sqrt(dx*dx + dy*dy)
                if dist > patch_size:
                    continue
                
                px, py = patch_x + dx, patch_y + dy
                if 0 <= px < size and 0 <= py < size:
                    fade = 1 - (dist / patch_size)
                    current = surface.get_at((px, py))
                    
                    # Mix with dirt color
                    dirt_color = (139, 69, 19)
                    new_color = (
                        int(current[0] * (1 - fade) + dirt_color[0] * fade),
                        int(current[1] * (1 - fade) + dirt_color[1] * fade),
                        int(current[2] * (1 - fade) + dirt_color[2] * fade),
                        255
                    )
                    surface.set_at((px, py), new_color)
    
    return surface

def main():
    pygame.init()
    pygame.font.init()
    
    # Set up the display
    screen = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
    pygame.display.set_caption("3D Flight Simulator")
    init_opengl(SCREEN_WIDTH, SCREEN_HEIGHT)
    
    # Load textures
    earth_surface = generate_earth_texture()
    earth_texture = load_texture_from_surface(earth_surface)
    
    cloud_surface = generate_cloud_texture()
    cloud_texture = load_texture_from_surface(cloud_surface)
    
    terrain_surface = generate_terrain_texture()
    terrain_texture = load_texture_from_surface(terrain_surface)
    
    # Generate terrain
    terrain_vertices, terrain_resolution = generate_terrain(size=40, resolution=50)
    
    # Create aircraft
    aircraft = Aircraft()
    aircraft.position = [0, 1, -5]  # Start position
    
    # Generate clouds
    clouds = []
    for _ in range(50):
        cloud_pos = [
            random.uniform(-20, 20),
            random.uniform(3, 10),
            random.uniform(-20, 20)
        ]
        cloud_size = random.uniform(0.5, 2.0)
        clouds.append(Cloud(cloud_pos, cloud_size))
    
    # Generate buildings for a city
    buildings = []
    for _ in range(30):
        building_pos = [
            random.uniform(-10, 10),
            0,
            random.uniform(-10, 10)
        ]
        
        # Make sure buildings aren't too close to the runway
        dist_to_runway = math.sqrt(building_pos[0]**2 + building_pos[2]**2)
        if dist_to_runway < 3:
            continue
        
        building_width = random.uniform(0.3, 1.2)
        building_height = random.uniform(0.5, 3.0)
        building_depth = random.uniform(0.3, 1.2)
        
        buildings.append(Building(building_pos, building_width, building_height, building_depth))
    
    # Clock for timing
    clock = pygame.time.Clock()
    fps_values = []
    fps_update_time = 0
    current_fps = 0
    
    # Main game loop
    running = True
    while running:
        # Calculate delta time
        dt = clock.tick(60) / 1000.0
        
        # Update FPS counter
        fps_values.append(1.0 / max(dt, 0.001))
        if len(fps_values) > 10:
            fps_values.pop(0)
        
        # Update FPS display every 0.5 seconds
        fps_update_time += dt
        if fps_update_time >= 0.5:
            current_fps = sum(fps_values) / len(fps_values)
            fps_update_time = 0
        
        # Process events
        for event in pygame.event.get():
            if event.type == QUIT:
                running = False
            elif event.type == KEYDOWN:
                if event.key == K_ESCAPE:
                    running = False
                elif event.key == K_c:
                    # Cycle through camera modes
                    aircraft.camera_mode = (aircraft.camera_mode + 1) % 3
                elif event.key == K_r:
                    # Reset aircraft position
                    aircraft.position = [0, 1, -5]
                    aircraft.velocity = [0, 0, 0]
                    aircraft.orientation = [0, 0, 0]
                    aircraft.throttle = 0.0
                elif event.key == K_f:
                    # Toggle FPS display
                    DEBUG_MODE = not DEBUG_MODE
        
        # Get keyboard state
        keys = pygame.key.get_pressed()
        
        # Flight controls
        if keys[K_w]:
            aircraft.pitch_input = -1.0  # Pitch down
        elif keys[K_s]:
            aircraft.pitch_input = 1.0   # Pitch up
        else:
            aircraft.pitch_input = 0.0
        
        if keys[K_a]:
            aircraft.roll_input = -1.0   # Roll left
        elif keys[K_d]:
            aircraft.roll_input = 1.0    # Roll right
        else:
            aircraft.roll_input = 0.0
        
        if keys[K_q]:
            aircraft.yaw_input = -1.0    # Yaw left
        elif keys[K_e]:
            aircraft.yaw_input = 1.0     # Yaw right
        else:
            aircraft.yaw_input = 0.0
        
        # Throttle control
        if keys[K_UP]:
            aircraft.throttle = min(1.0, aircraft.throttle + 0.01)
        elif keys[K_DOWN]:
            aircraft.throttle = max(0.0, aircraft.throttle - 0.01)
        
        # Speed boost with shift
        if keys[K_LSHIFT]:
            aircraft.max_speed = 1.0
        else:
            aircraft.max_speed = 0.5
        
        # Update aircraft physics
        aircraft.update(dt)
        
        # Set up the camera
        aircraft.setup_camera()
        
        # Clear the screen
        glClearColor(0.5, 0.7, 1.0, 1.0)  # Sky blue
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        
        # Draw sky
        draw_sky()
        
        # Draw terrain
        draw_terrain(terrain_vertices, terrain_resolution, terrain_texture)
        
        # Draw runway
        draw_runway()
        
        # Draw buildings
        for building in buildings:
            building.draw()
        
        # Draw aircraft (except in cockpit view)
        if aircraft.camera_mode != 0:
            aircraft.draw()
        
        # Draw clouds
        glBindTexture(GL_TEXTURE_2D, cloud_texture)
        for cloud in clouds:
            cloud.draw()
        
        # Draw the Earth in the distance
        draw_earth(earth_texture, [0, -100, 0], 100)
        
        # Swap buffers
        pygame.display.flip()
        
        # Draw HUD overlay (after OpenGL rendering)
        draw_surface = pygame.Surface((SCREEN_WIDTH, SCREEN_HEIGHT), pygame.SRCALPHA)
        draw_hud(draw_surface, aircraft, current_fps)
        
        # Temporarily disable OpenGL and draw the HUD
        pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT), DOUBLEBUF | OPENGL)
        screen.blit(draw_surface, (0, 0))
        pygame.display.flip()
    
    pygame.quit()

if __name__ == '__main__':
    main()