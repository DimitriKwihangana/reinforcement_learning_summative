import numpy as np
import random

# Constants for the grid
GRID_SIZE = 15
CELL_SIZE = 1.0
WALL_HEIGHT = 0.5
AGENT_SIZE = 0.4
VICTIM_SIZE = 0.3
BASE_SIZE = 2.0

# Colors
FLOOR_COLOR = (0.8, 0.8, 0.8, 1.0)  # Light gray
WALL_COLOR = (0.5, 0.5, 0.5, 1.0)    # Dark gray
CORRIDOR_COLOR = (0.7, 0.7, 0.9, 1.0)  # Light blue
ROOM_COLOR = (0.9, 0.8, 0.7, 1.0)    # Tan
OBSTACLE_COLOR = (0.3, 0.3, 0.3, 1.0)  # Darker gray
AGENT_COLOR = (0.0, 0.0, 1.0, 1.0)   # Blue
VICTIM_COLOR = (1.0, 0.5, 0.0, 1.0)  # Orange
BASE_COLOR = (0.0, 1.0, 0.0, 1.0)    # Green
FIRE_COLOR = (1.0, 0.3, 0.0, 0.7)    # Red-orange with transparency

# Environment representation
# 0 = corridor, 1 = room, 2 = wall, 3 = obstacle, 4 = base station
class DisasterEnvironment:
    def __init__(self, size=GRID_SIZE):
        self.size = size
        self.grid = np.ones((size, size), dtype=int) * 2  # Initialize all as walls
        self.agent_pos = [1, 1]  # Starting position
        self.victims = []
        self.fires = []
        self.base_station = [0, 0]  # Top-left corner
        
        # Create a simple building layout with rooms and corridors
        self.generate_environment()
        
    def generate_environment(self):
        # Create base station at top-left
        self.grid[0:2, 0:2] = 4  # Base station
        
        # Create main corridors
        for i in range(1, self.size-1):
            # Horizontal corridors
            if i % 4 == 0:
                self.grid[i, 1:self.size-1] = 0
            # Vertical corridors
            if i % 4 == 2:
                self.grid[1:self.size-1, i] = 0
        
        # Create rooms
        for i in range(2, self.size-2, 4):
            for j in range(2, self.size-2, 4):
                # Room area (3x3)
                self.grid[i:i+3, j:j+3] = 1
                
                # Add random obstacles in some rooms
                if random.random() < 0.5:
                    obstacle_x = random.randint(i, i+2)
                    obstacle_y = random.randint(j, j+2)
                    self.grid[obstacle_x, obstacle_y] = 3
                
                # Add victims in some rooms
                if random.random() < 0.7:
                    victim_x = random.choice([x for x in range(i, i+3) if self.grid[x, j+1] == 1])
                    victim_y = random.choice([y for y in range(j, j+3) if self.grid[i+1, y] == 1])
                    self.victims.append((victim_x, victim_y))
                
                # Add fire hazards in some rooms
                if random.random() < 0.6:
                    fire_x = random.choice([x for x in range(i, i+3) if self.grid[x, j+1] == 1])
                    fire_y = random.choice([y for y in range(j, j+3) if self.grid[i+1, y] == 1])
                    self.fires.append((fire_x, fire_y))
        
        # Connect rooms to corridors
        for i in range(2, self.size-2, 4):
            for j in range(2, self.size-2, 4):
                # Connect to horizontal corridor
                if i > 0:
                    door_y = random.randint(j, j+2)
                    self.grid[i-1, door_y] = 0  # Door
                
                # Connect to vertical corridor
                if j > 0:
                    door_x = random.randint(i, i+2)
                    self.grid[door_x, j-1] = 0  # Door
    
    # Add any additional environment methods here for agent training
    def step(self, action):
        # This is a placeholder for an RL interface
        # You'd implement movement, reward calculation, etc.
        pass
    
    def reset(self):
        # Reset the environment to initial state
        self.__init__(self.size)
        # Return initial observation
        return self.get_observation()
    
    def get_observation(self):
        # Return the current state observation
        # This is a placeholder - implement based on your needs
        return {
            'grid': self.grid.copy(),
            'agent_pos': self.agent_pos.copy(),
            'victims': self.victims.copy(),
            'fires': self.fires.copy()
        }