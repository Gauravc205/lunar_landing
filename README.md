Deep Q-Learning for Lunar Lander

This project implements a Deep Q-Network (DQN) to train an AI agent to land a spacecraft in the OpenAI Gymnasium "LunarLander-v3" environment. The agent learns through reinforcement learning by interacting with the environment and updating its policy based on rewards.

Installation
Ensure you have the required dependencies installed before running the project:
```python
!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]
!pip install torch numpy pygame imageio
