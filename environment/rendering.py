import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import imageio
import os
from PIL import Image
import time

from custom_env import DisasterEnvironment, GRID_SIZE, CELL_SIZE, WALL_HEIGHT, AGENT_SIZE
from custom_env import VICTIM_SIZE, FLOOR_COLOR, WALL_COLOR, CORRIDOR_COLOR, ROOM_COLOR
from custom_env import OBSTACLE_COLOR, AGENT_COLOR, VICTIM_COLOR, BASE_COLOR, FIRE_COLOR

# OpenGL helper functions
def draw_cube(x, y, z, size=1.0):
    vertices = [
        [x - size/2, y - size/2, z],
        [x + size/2, y - size/2, z],
        [x + size/2, y + size/2, z],
        [x - size/2, y + size/2, z],
        [x - size/2, y - size/2, z + size],
        [x + size/2, y - size/2, z + size],
        [x + size/2, y + size/2, z + size],
        [x - size/2, y + size/2, z + size]
    ]
    
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0),
        (4, 5), (5, 6), (6, 7), (7, 4),
        (0, 4), (1, 5), (2, 6), (3, 7)
    ]
    
    faces = [
        (0, 1, 2, 3),  # Bottom
        (4, 5, 6, 7),  # Top
        (0, 1, 5, 4),  # Front
        (2, 3, 7, 6),  # Back
        (0, 3, 7, 4),  # Left
        (1, 2, 6, 5)   # Right
    ]
    
    glBegin(GL_QUADS)
    for face in faces:
        for vertex in face:
            glVertex3fv(vertices[vertex])
    glEnd()

def draw_cylinder(x, y, z, radius=0.5, height=1.0, slices=20):
    glPushMatrix()
    glTranslatef(x, y, z)
    quadric = gluNewQuadric()
    gluCylinder(quadric, radius, radius, height, slices, 4)
    gluDisk(quadric, 0, radius, slices, 4)  # Bottom cap
    glTranslatef(0, 0, height)
    gluDisk(quadric, 0, radius, slices, 4)  # Top cap
    glPopMatrix()

def draw_victim(x, y):
    # Draw a simple humanoid figure
    glColor4fv(VICTIM_COLOR)
    # Body
    draw_cylinder(x, y, 0, VICTIM_SIZE/4, VICTIM_SIZE)
    # Head
    glPushMatrix()
    glTranslatef(x, y, VICTIM_SIZE)
    sphere = gluNewQuadric()
    gluSphere(sphere, VICTIM_SIZE/5, 10, 10)
    glPopMatrix()

def draw_fire(x, y, time_val):
    # Animate fire based on time
    size = 0.3 + 0.1 * np.sin(time_val * 5.0)
    height = 0.4 + 0.2 * np.sin(time_val * 7.0)
    
    # Fire base
    glColor4f(1.0, 0.3, 0.0, 0.7)  # Red-orange with transparency
    draw_cylinder(x, y, 0, size, height/2, 8)
    
    # Fire top
    glColor4f(1.0, 0.7, 0.0, 0.5)  # Yellow-orange with more transparency
    draw_cylinder(x, y, height/2, size/2, height/2, 8)

def render_environment(env, time_val):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    # Reset the view
    glLoadIdentity()
    
    # Position the camera for an isometric view
    gluLookAt(
        GRID_SIZE * 0.8, -GRID_SIZE * 0.8, GRID_SIZE * 0.8,  # Camera position
        GRID_SIZE/2, GRID_SIZE/2, 0,  # Look at position
        0, 0, 1  # Up vector
    )
    
    # Draw the grid
    for i in range(env.size):
        for j in range(env.size):
            cell_type = env.grid[i, j]
            cell_x = i * CELL_SIZE
            cell_y = j * CELL_SIZE
            
            # Draw floor for all cells
            glColor4fv(FLOOR_COLOR)
            glBegin(GL_QUADS)
            glVertex3f(cell_x, cell_y, 0)
            glVertex3f(cell_x + CELL_SIZE, cell_y, 0)
            glVertex3f(cell_x + CELL_SIZE, cell_y + CELL_SIZE, 0)
            glVertex3f(cell_x, cell_y + CELL_SIZE, 0)
            glEnd()
            
            if cell_type == 0:  # Corridor
                glColor4fv(CORRIDOR_COLOR)
                glBegin(GL_QUADS)
                glVertex3f(cell_x, cell_y, 0.01)  # Slightly above floor
                glVertex3f(cell_x + CELL_SIZE, cell_y, 0.01)
                glVertex3f(cell_x + CELL_SIZE, cell_y + CELL_SIZE, 0.01)
                glVertex3f(cell_x, cell_y + CELL_SIZE, 0.01)
                glEnd()
            
            elif cell_type == 1:  # Room
                glColor4fv(ROOM_COLOR)
                glBegin(GL_QUADS)
                glVertex3f(cell_x, cell_y, 0.01)
                glVertex3f(cell_x + CELL_SIZE, cell_y, 0.01)
                glVertex3f(cell_x + CELL_SIZE, cell_y + CELL_SIZE, 0.01)
                glVertex3f(cell_x, cell_y + CELL_SIZE, 0.01)
                glEnd()
            
            elif cell_type == 2:  # Wall
                glColor4fv(WALL_COLOR)
                draw_cube(cell_x + CELL_SIZE/2, cell_y + CELL_SIZE/2, WALL_HEIGHT/2, CELL_SIZE)
            
            elif cell_type == 3:  # Obstacle
                glColor4fv(OBSTACLE_COLOR)
                draw_cube(cell_x + CELL_SIZE/2, cell_y + CELL_SIZE/2, WALL_HEIGHT/2, CELL_SIZE * 0.8)
            
            elif cell_type == 4:  # Base station
                glColor4fv(BASE_COLOR)
                glBegin(GL_QUADS)
                glVertex3f(cell_x, cell_y, 0.02)  # Slightly above rooms
                glVertex3f(cell_x + CELL_SIZE, cell_y, 0.02)
                glVertex3f(cell_x + CELL_SIZE, cell_y + CELL_SIZE, 0.02)
                glVertex3f(cell_x, cell_y + CELL_SIZE, 0.02)
                glEnd()
    
    # Draw victims
    for victim_pos in env.victims:
        x, y = victim_pos
        draw_victim(x * CELL_SIZE + CELL_SIZE/2, y * CELL_SIZE + CELL_SIZE/2)
    
    # Draw fire hazards (animated)
    for fire_pos in env.fires:
        x, y = fire_pos
        draw_fire(x * CELL_SIZE + CELL_SIZE/2, y * CELL_SIZE + CELL_SIZE/2, time_val)
    
    # Draw agent (robot)
    glColor4fv(AGENT_COLOR)
    agent_x = env.agent_pos[0] * CELL_SIZE + CELL_SIZE/2
    agent_y = env.agent_pos[1] * CELL_SIZE + CELL_SIZE/2
    draw_cube(agent_x, agent_y, AGENT_SIZE/2 + 0.05, AGENT_SIZE)  # Slightly above floor

def main():
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption("Disaster Response Robot Environment")
    
    # Set up the OpenGL environment
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    # Set up the perspective
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create the environment
    env = DisasterEnvironment()
    
    # Set up for GIF capture
    gif_frames = []
    frame_count = 0
    total_frames = 90  # 3 seconds at 30fps
    
    clock = pygame.time.Clock()
    running = True
    start_time = time.time()
    
    while running and frame_count < total_frames:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
        
        current_time = time.time() - start_time
        
        # Render the environment
        render_environment(env, current_time)
        
        # Capture frame for GIF
        buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
        image = Image.frombytes("RGB", display[0:2], buffer)
        image = image.transpose(Image.FLIP_TOP_BOTTOM)
        gif_frames.append(image)
        
        pygame.display.flip()
        clock.tick(30)
        frame_count += 1
    
    pygame.quit()
    
    # Save the GIF
    gif_path = "disaster_environment.gif"
    print(f"Saving GIF to {gif_path}...")
    gif_frames[0].save(
        gif_path,
        save_all=True,
        append_images=gif_frames[1:],
        duration=33,  # milliseconds per frame
        loop=0  # 0 means loop forever
    )
    print("GIF saved successfully!")

if __name__ == "__main__":
    main()