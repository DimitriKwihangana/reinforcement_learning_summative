import gymnasium as gym
import numpy as np
from gymnasium import spaces
from custom_env import DisasterEnvironment, GRID_SIZE

class DisasterEnvWrapper(gym.Env):
    """
    Wrapper to make the DisasterEnvironment compatible with Stable Baselines 3
    """
    metadata = {'render_modes': ['human']}
    
    def __init__(self, env):
        super(DisasterEnvWrapper, self).__init__()
        self.env = env
        self.size = env.size
        
        # Define action space (4 directions: up, right, down, left)
        self.action_space = spaces.Discrete(4)
        
        # Calculate the observation shape correctly
        # Grid (size*size) + agent_pos (2) + victim_vec (2) + fire_vec (2) + base_vec (2) + carrying (1)
        obs_dim = self.size * self.size + 2 + 2 + 2 + 2 + 1
        
        # Define observation space with the correct shape
        self.observation_space = spaces.Box(
            low=-np.inf,
            high=np.inf,
            shape=(obs_dim,),
            dtype=np.float32
        )
        
        # Set reward ranges and parameters
        self.step_penalty = -0.01
        self.victim_rescue_reward = 10.0
        self.fire_penalty = -5.0
        self.collision_penalty = -1.0
        self.return_to_base_reward = 5.0
        
        # Keep track of rescued victims
        self.rescued_victims = []
        self.carrying_victim = False
        
        # Verify observation shape matches what we expect
        test_obs = self._get_observation()
        assert test_obs.shape[0] == obs_dim, f"Observation shape mismatch: expected {obs_dim}, got {test_obs.shape[0]}"
        
    def reset(self, seed=None, options=None):
        # Reset the environment
        super().reset(seed=seed)
        self.env.reset()
        self.rescued_victims = []
        self.carrying_victim = False
        return self._get_observation(), {}
        
    def step(self, action):
        # Execute action in the environment
        # 0: Up, 1: Right, 2: Down, 3: Left
        new_pos = self.env.agent_pos.copy()
        
        if action == 0:  # Up
            new_pos[1] -= 1
        elif action == 1:  # Right
            new_pos[0] += 1
        elif action == 2:  # Down
            new_pos[1] += 1
        elif action == 3:  # Left
            new_pos[0] -= 1
            
        # Check if movement is valid (not into a wall)
        reward = self.step_penalty
        done = False
        truncated = False
        info = {}
        
        # Check for collision with wall or obstacle
        if (new_pos[0] < 0 or new_pos[0] >= self.size or 
            new_pos[1] < 0 or new_pos[1] >= self.size or
            self.env.grid[new_pos[0], new_pos[1]] in [2, 3]):  # Wall or obstacle
            reward += self.collision_penalty
        else:
            self.env.agent_pos = new_pos
            
            # Check for victim rescue
            pos_tuple = (self.env.agent_pos[0], self.env.agent_pos[1])
            
            # If at base station with victim
            if self.env.grid[pos_tuple[0], pos_tuple[1]] == 4 and self.carrying_victim:
                reward += self.return_to_base_reward
                self.carrying_victim = False
                self.rescued_victims.append(pos_tuple)
            
            # Check for victim pickup
            for victim in self.env.victims:
                if pos_tuple == victim and victim not in self.rescued_victims and not self.carrying_victim:
                    self.carrying_victim = True
                    reward += self.victim_rescue_reward / 2
                    break
            
            # Check for fire penalty
            for fire in self.env.fires:
                if pos_tuple == fire:
                    reward += self.fire_penalty
                    break
            
            # Check if all victims rescued
            if len(self.rescued_victims) == len(self.env.victims):
                done = True
                reward += 20.0  # Bonus for completing the mission
                
        # Get the next observation
        obs = self._get_observation()
        
        return obs, reward, done, truncated, info
        
    def _get_observation(self):
        """
        Convert the environment state to an observation compatible with SB3
        """
        # Flatten the grid
        grid_flat = self.env.grid.flatten() / 4.0  # Normalize values
        
        # Agent position
        agent_pos = np.array(self.env.agent_pos) / self.size  # Normalize to [0, 1]
        
        # Vector to nearest victim (if any)
        nearest_victim_vec = np.array([0.0, 0.0])
        min_dist = float('inf')
        for victim in self.env.victims:
            if victim not in self.rescued_victims:
                dist = np.linalg.norm(np.array(victim) - np.array(self.env.agent_pos))
                if dist < min_dist:
                    min_dist = dist
                    nearest_victim_vec = (np.array(victim) - np.array(self.env.agent_pos)) / self.size
        
        # Vector to nearest fire (if any)
        nearest_fire_vec = np.array([0.0, 0.0])
        min_dist = float('inf')
        for fire in self.env.fires:
            dist = np.linalg.norm(np.array(fire) - np.array(self.env.agent_pos))
            if dist < min_dist:
                min_dist = dist
                nearest_fire_vec = (np.array(fire) - np.array(self.env.agent_pos)) / self.size
        
        # Vector to base station
        base_vec = (np.array(self.env.base_station) - np.array(self.env.agent_pos)) / self.size
        
        # Combine observations
        carrying_victim = np.array([1.0]) if self.carrying_victim else np.array([0.0])
        
        obs = np.concatenate([
            grid_flat, 
            agent_pos, 
            nearest_victim_vec, 
            nearest_fire_vec, 
            base_vec,
            carrying_victim
        ])
        
        return obs.astype(np.float32)
        
    def render(self):
        # Use the existing rendering if available, otherwise do nothing
        # This assumes you'll use the separate rendering.py for visualization
        pass