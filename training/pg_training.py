# Training script for PPO using Stable Baselines 3
import os
import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
from custom_env import DisasterEnvironment
from sb3_env_wrapper import DisasterEnvWrapper

def make_env():
    """
    Helper function to create a vectorized environment
    """
    def _init():
        env = DisasterEnvironment()
        env = DisasterEnvWrapper(env)
        return env
    return _init

def train_ppo(total_timesteps=100000):
    """
    Train a PPO agent on the disaster environment
    
    Args:
        total_timesteps (int): Number of timesteps to train for
        
    Returns:
        The trained PPO model
    """
    print("Creating disaster environment for PPO training...")
    
    # Create vectorized environment (4 parallel environments)
    env = SubprocVecEnv([make_env() for _ in range(4)])
    
    print(f"Starting PPO training for {total_timesteps} timesteps...")
    
    # Create and train the PPO model
    model = PPO(
        "MlpPolicy",
        env,
        learning_rate=3e-4,
        n_steps=2048,
        batch_size=32,
        n_epochs=10,
        gamma=0.97,
        gae_lambda=0.70,
        clip_range=0.2,
        ent_coef=0.0,
        verbose=1
    )
    
    model.learn(total_timesteps=total_timesteps)
    
    # Save the trained model
    model_path = os.path.join("models", "ppo_disaster_robot")
    model.save(model_path)
    print(f"PPO model saved to {model_path}")
    
    return model

def evaluate_ppo(model=None, num_episodes=10, render=False):
    """
    Evaluate a trained PPO model
    
    Args:
        model: The PPO model to evaluate (if None, loads from saved file)
        num_episodes (int): Number of episodes to evaluate
        render (bool): Whether to render the environment during evaluation
    
    Returns:
        Mean and std of episode rewards
    """
    # Load model if not provided
    if model is None:
        model_path = os.path.join("models", "ppo_disaster_robot")
        model = PPO.load(model_path)
        print(f"Loaded PPO model from {model_path}")
    
    # Create environment for evaluation
    env = DisasterEnvironment()
    env = DisasterEnvWrapper(env)
    
    # Run evaluation
    episode_rewards = []
    for i in range(num_episodes):
        obs = env.reset()
        done = False
        total_reward = 0
        steps = 0
        
        while not done:
            action, _states = model.predict(obs, deterministic=True)
            obs, reward, done, info = env.step(action)
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
    model = train_ppo(total_timesteps=100000)
    evaluate_ppo(model, render=True)