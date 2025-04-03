import pygame
from pygame.locals import *
from OpenGL.GL import *
from OpenGL.GLU import *
import numpy as np
import time
from PIL import Image
import os
import sys

# Import your custom environment
from custom_env import DisasterEnvironment, GRID_SIZE, CELL_SIZE, WALL_HEIGHT, AGENT_SIZE
from custom_env import VICTIM_SIZE, FLOOR_COLOR, WALL_COLOR, CORRIDOR_COLOR, ROOM_COLOR
from custom_env import OBSTACLE_COLOR, AGENT_COLOR, VICTIM_COLOR, BASE_COLOR, FIRE_COLOR

# Import your trained model - adjust import path based on your project structure
# Example if using a DQN model
sys.path.append('.')  # Add the current directory to path
try:
    # Try to import your model - adjust these imports based on your project
    from models.dqn_disaster_robot import DQNAgent
    model_path = "models/ppo_disaster_robot.zip"
    model_type = "dqn"
except ImportError:
    try:
        # Try policy gradient model if DQN fails
        from models.pg_disaster_robot import PGAgent
        model_path = "models/pg_disaster_robot.zip"
        model_type = "pg"
    except ImportError:
        print("Could not import trained model. Running with random actions instead.")
        model_type = "random"

# OpenGL rendering functions from rendering.py
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
    glColor4fv(VICTIM_COLOR)
    draw_cylinder(x, y, 0, VICTIM_SIZE/4, VICTIM_SIZE)
    glPushMatrix()
    glTranslatef(x, y, VICTIM_SIZE)
    sphere = gluNewQuadric()
    gluSphere(sphere, VICTIM_SIZE/5, 10, 10)
    glPopMatrix()

def draw_fire(x, y, time_val):
    size = 0.3 + 0.1 * np.sin(time_val * 5.0)
    height = 0.4 + 0.2 * np.sin(time_val * 7.0)
    
    glColor4f(1.0, 0.3, 0.0, 0.7)
    draw_cylinder(x, y, 0, size, height/2, 8)
    
    glColor4f(1.0, 0.7, 0.0, 0.5)
    draw_cylinder(x, y, height/2, size/2, height/2, 8)

def render_environment(env, time_val):
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    
    glLoadIdentity()
    
    gluLookAt(
        GRID_SIZE * 0.8, -GRID_SIZE * 0.8, GRID_SIZE * 0.8,
        GRID_SIZE/2, GRID_SIZE/2, 0,
        0, 0, 1
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
                glVertex3f(cell_x, cell_y, 0.01)
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
                glVertex3f(cell_x, cell_y, 0.02)
                glVertex3f(cell_x + CELL_SIZE, cell_y, 0.02)
                glVertex3f(cell_x + CELL_SIZE, cell_y + CELL_SIZE, 0.02)
                glVertex3f(cell_x, cell_y + CELL_SIZE, 0.02)
                glEnd()
    
    # Draw victims
    for victim_pos in env.victims:
        x, y = victim_pos
        draw_victim(x * CELL_SIZE + CELL_SIZE/2, y * CELL_SIZE + CELL_SIZE/2)
    
    # Draw fire hazards
    for fire_pos in env.fires:
        x, y = fire_pos
        draw_fire(x * CELL_SIZE + CELL_SIZE/2, y * CELL_SIZE + CELL_SIZE/2, time_val)
    
    # Draw agent
    glColor4fv(AGENT_COLOR)
    agent_x = env.agent_pos[0] * CELL_SIZE + CELL_SIZE/2
    agent_y = env.agent_pos[1] * CELL_SIZE + CELL_SIZE/2
    draw_cube(agent_x, agent_y, AGENT_SIZE/2 + 0.05, AGENT_SIZE)

# Extend DisasterEnvironment to implement the step method if not already done
class ExtendedDisasterEnvironment(DisasterEnvironment):
    def __init__(self, size=GRID_SIZE):
        super().__init__(size)
        self.found_victims = []
        self.visited_cells = set([(self.agent_pos[0], self.agent_pos[1])])
        self.done = False
        self.steps = 0
        self.max_steps = 200  # Limit episode length
        self.episode_reward = 0  # Track episode rewards
        
    def step(self, action):
        """
        Execute action in the environment.
        Actions: 0=up, 1=right, 2=down, 3=left
        """
        self.steps += 1
        reward = -0.01  # Small penalty for each step
        
        # Move agent based on action
        old_pos = self.agent_pos.copy()
        
        if action == 0 and self.agent_pos[0] > 0:  # Up
            self.agent_pos[0] -= 1
        elif action == 1 and self.agent_pos[1] < self.size - 1:  # Right
            self.agent_pos[1] += 1
        elif action == 2 and self.agent_pos[0] < self.size - 1:  # Down
            self.agent_pos[0] += 1
        elif action == 3 and self.agent_pos[1] > 0:  # Left
            self.agent_pos[1] -= 1
        
        # Check if new position is valid (not a wall or obstacle)
        new_cell = self.grid[self.agent_pos[0], self.agent_pos[1]]
        if new_cell == 2 or new_cell == 3:  # Wall or obstacle
            self.agent_pos = old_pos  # Revert movement
            reward -= 0.5  # Penalty for hitting wall/obstacle
        
        # Add to visited cells
        self.visited_cells.add((self.agent_pos[0], self.agent_pos[1]))
        
        # Check for victim rescue
        for victim in self.victims:
            if tuple(self.agent_pos) == victim and victim not in self.found_victims:
                self.found_victims.append(victim)
                reward += 50.0  # Big reward for finding victim
        
        # Check for fire (negative reward)
        for fire in self.fires:
            if tuple(self.agent_pos) == fire:
                reward -= 3.0  # Penalty for entering fire
        
        # Check if at base station
        if self.grid[self.agent_pos[0], self.agent_pos[1]] == 4 and len(self.found_victims) > 0:
            reward += 5.0 * len(self.found_victims)  # Bonus for returning victims to base
            self.found_victims = []  # Reset found victims
        
        # Check for episode end
        if len(self.found_victims) == len(self.victims):
            self.done = True
            reward += 50.0  # Big bonus for finding all victims
        
        if self.steps >= self.max_steps:
            self.done = True
        
        # Update episode reward
        self.episode_reward += reward
        
        # Return SARS tuple
        return self.get_observation(), reward, self.done, {}
    
    def reset(self):
        # Reset environment and initialize a new episode
        super().reset()
        self.found_victims = []
        self.visited_cells = set([(self.agent_pos[0], self.agent_pos[1])])
        self.done = False
        self.steps = 0
        self.episode_reward = 0  # Reset episode reward
        return self.get_observation()
    
    def get_observation(self):
        """Create a simple grid-based observation"""
        # For simplicity, return a flattened view of the nearby grid
        # Real implementation would depend on how your model expects observations
        view_size = 3  # Agent can see 3x3 grid around it
        
        # Initialize with walls (padding)
        view = np.ones((view_size, view_size), dtype=int) * 2
        
        # Fill in the observable area
        for i in range(view_size):
            for j in range(view_size):
                x = self.agent_pos[0] - 1 + i
                y = self.agent_pos[1] - 1 + j
                
                if 0 <= x < self.size and 0 <= y < self.size:
                    view[i, j] = self.grid[x, y]
        
        # Add indicators for victims and fires in view
        victim_indicator = np.zeros((view_size, view_size), dtype=int)
        fire_indicator = np.zeros((view_size, view_size), dtype=int)
        
        for victim in self.victims:
            x, y = victim
            if (abs(x - self.agent_pos[0]) <= 1 and 
                abs(y - self.agent_pos[1]) <= 1):
                victim_indicator[x - self.agent_pos[0] + 1, y - self.agent_pos[1] + 1] = 1
        
        for fire in self.fires:
            x, y = fire
            if (abs(x - self.agent_pos[0]) <= 1 and 
                abs(y - self.agent_pos[1]) <= 1):
                fire_indicator[x - self.agent_pos[0] + 1, y - self.agent_pos[1] + 1] = 1
        
        # Combine all information into one observation
        obs = {
            'grid': view.flatten(),
            'victims': victim_indicator.flatten(),
            'fires': fire_indicator.flatten(),
            'agent_pos': np.array(self.agent_pos) / self.size,  # Normalized position
            'num_found_victims': len(self.found_victims),
            'num_total_victims': len(self.victims)
        }
        
        # For compatibility with models that expect a simple array
        # Adjust based on your model's input requirements
        flat_obs = np.concatenate([
            view.flatten(), 
            victim_indicator.flatten(), 
            fire_indicator.flatten(),
            np.array(self.agent_pos) / self.size,
            np.array([len(self.found_victims), len(self.victims)]) / max(1, len(self.victims))
        ])
        
        return flat_obs

# Random agent for fallback
class RandomAgent:
    def __init__(self, env):
        self.action_space = 4  # Up, Right, Down, Left
    
    def predict(self, observation):
        return np.random.randint(0, self.action_space)
    
    def load(self, path):
        print(f"Using random agent instead of loading from {path}")

def main():
    # Allow command line arguments for max episodes
    max_episodes = 2  # Default to 3 episodes
    if len(sys.argv) > 1:
        try:
            max_episodes = int(sys.argv[1])
            print(f"Setting maximum episodes to: {max_episodes}")
        except ValueError:
            print(f"Invalid number of episodes. Using default: {max_episodes}")
    
    pygame.init()
    display = (800, 600)
    pygame.display.set_mode(display, DOUBLEBUF | OPENGL)
    pygame.display.set_caption(f"Trained Disaster Response Robot (Max Episodes: {max_episodes})")
    
    # Set up OpenGL
    glEnable(GL_DEPTH_TEST)
    glEnable(GL_BLEND)
    glBlendFunc(GL_SRC_ALPHA, GL_ONE_MINUS_SRC_ALPHA)
    
    glMatrixMode(GL_PROJECTION)
    gluPerspective(45, (display[0] / display[1]), 0.1, 50.0)
    glMatrixMode(GL_MODELVIEW)
    
    # Create the environment using the extended class
    env = ExtendedDisasterEnvironment()
    
    # Initialize agent based on available model
    if model_type == "dqn":
        agent = DQNAgent(env)
        agent.load(model_path)
    elif model_type == "pg":
        agent = PGAgent(env)
        agent.load(model_path)
    else:
        agent = RandomAgent(env)
    
    # Set up for capturing frames
    save_gif = True
    gif_frames = []
    
    # Main loop variables
    clock = pygame.time.Clock()
    running = True
    start_time = time.time()
    
    # Reset environment to start
    observation = env.reset()
    frame_count = 0
    action_delay = 10  # Only take action every N frames
    
    # Add episode counter and reward tracking
    current_episode = 1
    episode_rewards = []
    
    try:
        while running and current_episode <= max_episodes:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
            
            current_time = time.time() - start_time
            
            # Take action periodically to slow down visualization
            if frame_count % action_delay == 0:
                # Get action from agent
                action = agent.predict(observation)
                
                # Execute action in environment
                observation, reward, done, _ = env.step(action)
                
                print(f"Episode: {current_episode}/{max_episodes}, Step: {env.steps}, Action: {action}, Reward: {reward:.2f}, "
                      f"Agent Pos: {env.agent_pos}, Victims Found: {len(env.found_victims)}/{len(env.victims)}")
                
                if done:
                    # Store episode reward
                    episode_rewards.append(env.episode_reward)
                    
                    # Print episode summary
                    print(f"\n==== Episode {current_episode} Complete ====")
                    print(f"Total Steps: {env.steps}")
                    print(f"Episode Reward: {env.episode_reward:.2f}")
                    print(f"Victims Rescued: {len(env.victims) - len(env.found_victims)}/{len(env.victims)}")
                    
                    if env.steps >= env.max_steps:
                        print("Maximum steps reached.")
                    else:
                        print("All victims rescued!")
                    print("==============================\n")
                    
                    current_episode += 1
                    if current_episode <= max_episodes:
                        # Reset environment for next episode
                        observation = env.reset()
                    else:
                        print(f"Reached maximum number of episodes ({max_episodes}). Quitting...")
                        
                        # Print overall performance summary
                        print("\n====== SIMULATION SUMMARY ======")
                        for i, reward in enumerate(episode_rewards):
                            print(f"Episode {i+1} Reward: {reward:.2f}")
                        print(f"Total Reward Across All Episodes: {sum(episode_rewards):.2f}")
                        print(f"Average Reward Per Episode: {sum(episode_rewards)/len(episode_rewards):.2f}")
                        print("===============================\n")
                        
                        running = False
            
            # Render the environment
            render_environment(env, current_time)
            
            # Update window title with episode info
            pygame.display.set_caption(f"Disaster Response Robot - Episode: {current_episode}/{max_episodes}")
            
            # Capture frame for GIF if enabled
            if save_gif:
                buffer = glReadPixels(0, 0, display[0], display[1], GL_RGB, GL_UNSIGNED_BYTE)
                image = Image.frombytes("RGB", display[0:2], buffer)
                image = image.transpose(Image.FLIP_TOP_BOTTOM)
                gif_frames.append(image)
            
            pygame.display.flip()
            clock.tick(30)  # Cap at 30 FPS
            frame_count += 1
            
    finally:
        pygame.quit()
        
        # If simulation ended early, print summary if we have any data
        if episode_rewards and not running:
            print("\n====== SIMULATION SUMMARY ======")
            for i, reward in enumerate(episode_rewards):
                print(f"Episode {i+1} Reward: {reward:.2f}")
            print(f"Total Reward Across All Episodes: {sum(episode_rewards):.2f}")
            print(f"Average Reward Per Episode: {sum(episode_rewards)/len(episode_rewards):.2f}")
            print("===============================\n")
        
        # Save GIF if frames were captured
        if save_gif and gif_frames:
            gif_path = f"disaster_agent_{len(episode_rewards)}_episodes.gif"
            print(f"Saving GIF to {gif_path}...")
            gif_frames[0].save(
                gif_path,
                save_all=True,
                append_images=gif_frames[1:300],  # Limit to first 300 frames to keep file size reasonable
                duration=33,  # 30 FPS
                loop=0  # Loop forever
            )
            print("GIF saved successfully!")

if __name__ == "__main__":
    main()