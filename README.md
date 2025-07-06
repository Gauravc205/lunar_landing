Deep Q-Learning for Lunar Lander
This project implements a Deep Q-Network (DQN) to train an AI agent to land a spacecraft in the OpenAI Gymnasium "LunarLander-v3" environment. The agent learns through reinforcement learning by interacting with the environment and updating its policy based on rewards.

Installation
Ensure you have the required dependencies installed before running the project:

!pip install gymnasium
!pip install "gymnasium[atari, accept-rom-license]"
!apt-get install -y swig
!pip install gymnasium[box2d]
!pip install torch numpy pygame imageio
Project Structure
dqn_lunar_lander.py - Main script implementing the DQN agent and training loop.
checkpoint.pth - Saved model weights after training.
README.md - Documentation for the project.
video.mp4 - Sample video of the trained agent in action.
Training the Agent
Environment Setup
The Lunar Lander environment consists of:

State space: 8-dimensional vector describing the lander’s position, velocity, angle, and other parameters.
Action space: 4 discrete actions (do nothing, fire left engine, fire main engine, fire right engine).
Rewards:
+100 to +200 for a successful landing.
-100 for crashing.
Small penalties for using fuel and unstable movement.
Neural Network Architecture
A fully connected neural network with two hidden layers:

Input: state_size (8)
Hidden layers: 64 neurons each
Output: action_size (4) with Q-values for each action
class Network(nn.Module):
    def __init__(self, state_size, action_size, seed=42):
        super(Network, self).__init__()
        self.fc1 = nn.Linear(state_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, action_size)
    
    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)
Training Loop
The agent is trained using experience replay and the Bellman equation:

for episode in range(1, num_episodes + 1):
    state, _ = env.reset()
    score = 0
    for t in range(max_timesteps):
        action = agent.act(state, epsilon)
        next_state, reward, done, _, _ = env.step(action)
        agent.step(state, action, reward, next_state, done)
        state = next_state
        score += reward
        if done:
            break
    scores.append(score)
    epsilon = max(epsilon_min, epsilon_decay * epsilon)
The model reaches a successful landing average score (>=200) in ~1300 episodes.

Running the Model
To visualize the trained agent:

show_video_of_model(agent, 'LunarLander-v3')
Results
Average Score Progression:
Episode 100: -157.75
Episode 500: -22.22
Episode 1000: 95.69
Episode 1335: 201.03 (Environment Solved!)
Saved Model: The trained agent's weights are stored.
Future Improvements
Implement Double DQN to reduce overestimation bias.
Use Dueling DQN to better distinguish value and advantage functions.
Experiment with Prioritized Experience Replay for more efficient learning.
References
OpenAI Gymnasium: https://gymnasium.farama.org/
Deep Q-Learning: https://www.cs.toronto.edu/~vmnih/docs/dqn.pdf
