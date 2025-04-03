import os
import argparse
from training import dqn_training, pg_training

def main():
    # Create command-line argument parser
    parser = argparse.ArgumentParser(description='Train reinforcement learning agents for disaster response')
    parser.add_argument('--algorithm', type=str, choices=['dqn', 'ppo'], default='dqn',
                      help='Algorithm to train (dqn or ppo)')
    parser.add_argument('--timesteps', type=int, default=100000,
                      help='Number of timesteps to train for')
    parser.add_argument('--eval', action='store_true',
                      help='Evaluate the model after training')
    parser.add_argument('--render', action='store_true',
                      help='Render the environment during evaluation')
    args = parser.parse_args()
    
    # Create models directory if it doesn't exist
    os.makedirs('models', exist_ok=True)
    
    # Train the selected algorithm
    if args.algorithm == 'dqn':
        model = dqn_training.train_dqn(total_timesteps=args.timesteps)
        if args.eval:
            dqn_training.evaluate_dqn(model, render=args.render)
    elif args.algorithm == 'ppo':
        model = pg_training.train_ppo(total_timesteps=args.timesteps)
        if args.eval:
            pg_training.evaluate_ppo(model, render=args.render)

if __name__ == "__main__":
    main()