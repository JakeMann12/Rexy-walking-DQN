import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import random
import gym
import copy
from tqdm import tqdm
from simple_rexy.envs.rexy_env import SimpleRexyEnv

# Define the neural network for DQN
class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 128)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(128, 256)
        self.fc2.weight.data.normal_(0, 0.1)
        self.fc3 = nn.Linear(256, 128)
        self.fc3.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(128, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        action_values = self.out(x)
        return action_values

# Define the DQN agent
class DQNAgent:
    def __init__(self, env, BATCH_SIZE=128, LR=0.01, GAMMA=0.90, 
                 EPSILON=0.9, EPSILON_DECAY=0.995, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000, 
                 Q_NETWORK_ITERATION=100):
        
        #Cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"GPU {torch.cuda.get_device_name(0)} is available.")
        else:
            self.device = torch.device("cpu")
            print("GPU is not available. Using CPU.")
        self.device = torch.device("cpu")

        # ESTABLISH REXY ENV
        self.env = gym.make(env).unwrapped
        self.num_actions = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0]

        # Q-Networks
        self.q_network = Net(self.num_states, self.num_actions).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=MEMORY_CAPACITY)#.to(self.device)

        # Batch, Gamma
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.q_network_iteration = Q_NETWORK_ITERATION

        # Epsilon-greedy strategy
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        # Counter for Q-network updates
        self.update_counter = 0

    def select_action(self, state):
        """
        Chooses action based on epsilon-greedy method
        """
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()  # RANDOM ACTION
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state)
                action = q_values.cpu().numpy()

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states to a tensor
        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-learning update
        q_values = self.q_network(states)
        target_q_values = self.target_network(next_states).detach()
        target = rewards + self.gamma * (1 - dones) * target_q_values.max(dim=1)[0]
        target = target.unsqueeze(1).expand_as(q_values)
        loss = self.loss_fn(q_values, target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.q_network_iteration == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, num_episodes, plot = False, save=True):
        episode_rewards = []
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                total_reward += reward
                state = next_state

                # NOTE: moved Decay epsilon into each step instead of in the replay funct.
                self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
                
                if done:
                    break

            #NOTE: put this back in every episode if it takes too long time-wise to learn.
            self.replay(self.batch_size)

            # Append episode reward to the list
            episode_rewards.append(total_reward)

            if episode == num_episodes:
                print(f"After {episode} episodes, Total Reward: {total_reward:.2f}")

        # After training, plot and save the static results
        if plot:
            self.plot_rewards(episode_rewards)

        if save:
            print("boutta save the model")
            self.save_model()

        return episode_rewards

    def save_model(self, model_path="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved! to {model_path}")

    def load_model(self, model_path="dqn_model.pth"):
        if os.path.exists(model_path):
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )  # Ensure consistency with target network
            print("Model loaded successfully.")
        else:
            print("Agent Model file not found.")
