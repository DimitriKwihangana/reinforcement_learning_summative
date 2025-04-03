# Disaster Response Reinforcement Learning Project

## Project Overview
This project focuses on developing and evaluating reinforcement learning (RL) models for a disaster response scenario. The goal is to train an agent to navigate through a disaster-stricken environment, avoiding hazards and rescuing victims. The project implements **Deep Q-Network (DQN)** and **Proximal Policy Optimization (PPO)** algorithms to optimize navigation and rescue efficiency.

## Folder Structure
```
WAREHOUSE_STORAGE_RL
│-- env/
│-- environment/
│   │-- images/
│   │   │-- disaster_agent_*.png
│   │-- logs/
│-- models/
│   │-- custom_env.py
│   │-- mainn.py
│   │-- rendering.py
│   │-- sb3_env_wrapper.py
│-- training/
│   │-- dqn_training.py
│   │-- pg_training.py
│   │-- main.py
│-- README.md
│-- requirements.txt
```

## Installation

### Prerequisites
Ensure you have the following dependencies installed:
- Python 3.8+
- TensorFlow or PyTorch
- OpenAI Gym
- NumPy
- Matplotlib

### Installation Steps
1. **Clone the repository:**
   ```sh
   git clone <repository-url>
   cd disaster-rl
   ```
2. **Create a virtual environment:**
   ```sh
   python -m venv env
   source env/bin/activate  # On Windows use: env\Scripts\activate
   ```
3. **Install dependencies:**
   ```sh
   pip install -r requirements.txt
   ```
4. **Run the training DQN model:**
   ```sh
   python training/dqn_training.py
   ```
5. **Run the training DQN model:**
   ```sh
   python training/pg_training.py
   ```



