import os
import numpy as np
from stable_baselines3 import DQN
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback
from custom_env import DisasterEnvironment
from sb3_env_wrapper import DisasterEnvWrapper


class LossLoggerCallback(BaseCallback):
    """
    Custom callback to log loss values during DQN training
    """
    def __init__(self, verbose=0):
        super(LossLoggerCallback, self).__init__(verbose)
        self.losses = []

    def _on_step(self) -> bool:
        if "loss" in self.model.logger.name_to_value:
            loss = self.model.logger.name_to_value["loss"]
            self.losses.append(loss)
            print(f"Step {self.num_timesteps}: Loss = {loss}")
        return True  # Continue training


def train_dqn(total_timesteps=100000):
    """
    Train a DQN agent and log training stability data (loss values)
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        
    Returns:
        The trained DQN model
    """
    print("Creating disaster environment for DQN training...")
    
    # Create and wrap the environment
    env = DisasterEnvironment()
    env = DisasterEnvWrapper(env)
    env = Monitor(env)
    
    # Print observation space shape for debugging
    print(f"Observation space shape: {env.observation_space.shape}")
    test_obs, _ = env.reset()
    print(f"Actual observation shape: {test_obs.shape}")
    
    print(f"Starting DQN training for {total_timesteps} timesteps...")
    
    # Create and train the DQN model
    model = DQN(
        "MlpPolicy",
        env,
        learning_rate=0.0001,
        buffer_size=100000,
        learning_starts=1000,
        batch_size=64,
        tau=0.1,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        target_update_interval=1000,
        exploration_fraction=0.1,
        exploration_initial_eps=1.0,
        exploration_final_eps=0.05,
        verbose=1
    )
    
    # Initialize loss logger callback
    loss_logger = LossLoggerCallback()

    # Train with callback
    model.learn(total_timesteps=total_timesteps, callback=loss_logger)

    # Save loss values for analysis
    loss_file = "dqn_loss_values.csv"
    np.savetxt(loss_file, loss_logger.losses, delimiter=",")
    print(f"Loss values saved to {loss_file}")

    # Save the trained model
    model_path = os.path.join("models", "dqn_disaster_robot")
    model.save(model_path)
    print(f"DQN model saved to {model_path}")
    
    return model


def evaluate_dqn(model=None, num_episodes=10, render=False):
    """
    Evaluate a trained DQN model
    
    Args:
        model: The DQN model to evaluate (if None, loads from saved file)
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment during evaluation
    
    Returns:
        Mean and std of episode rewards
    """
    # Load model if not provided
    if model is None:
        model_path = os.path.join("models", "dqn_disaster_robot")
        model = DQN.load(model_path)
        print(f"Loaded DQN model from {model_path}")
    
    # Create environment for evaluation
    env = DisasterEnvironment()
    env = DisasterEnvWrapper(env)
    
    # Run evaluation
    episode_rewards = []
    for i in range(num_episodes):
        obs, _ = env.reset()
        done = False
        truncated = False
        total_reward = 0
        steps = 0
        
        while not done and not truncated:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, truncated, info = env.step(action)
            total_reward += reward
            steps += 1
            
            if render:
                env.render()
        
        episode_rewards.append(total_reward)
        print(f"Episode {i+1}/{num_episodes}: Reward = {total_reward}, Steps = {steps}")
    
    mean_reward = np.mean(episode_rewards)
    std_reward = np.std(episode_rewards)
    print(f"Evaluation over {num_episodes} episodes: Mean reward = {mean_reward:.2f} ± {std_reward:.2f}")
    
    return mean_reward, std_reward


if __name__ == "__main__":
    # If run directly, train and evaluate the model
    model = train_dqn(total_timesteps=100000)
    evaluate_dqn(model, render=True)
