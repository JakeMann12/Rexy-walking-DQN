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
        self.fc1 = nn.Linear(num_states, 50)
        self.fc1.weight.data.normal_(0, 0.1)
        self.fc2 = nn.Linear(50, 30)
        self.fc2.weight.data.normal_(0, 0.1)
        self.out = nn.Linear(30, num_actions)
        self.out.weight.data.normal_(0, 0.1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        action_values = self.out(x)
        return action_values


# Define the DQN agent
class DQNAgent:
    def __init__(self, env, BATCH_SIZE = 128, LR = 0.01, GAMMA = 0.90, 
                 EPSILON = 0.9, EPSILON_DECAY = 0.995, EPSILON_MIN = 0.01, MEMORY_CAPACITY = 2000, 
                 Q_NETWORK_ITERATION = 100):
        # ESTABLISH REXY ENV
        self.env = gym.make(env).unwrapped
        self.num_actions = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0]

        # Q-Networks
        self.q_network = Net(self.num_states, self.num_actions)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=MEMORY_CAPACITY)

        #Batch, Gamma
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.q_network_iteration = Q_NETWORK_ITERATION

        # Epsilon-greedy strategy
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        # Counter for Q-network updates
        self.update_counter = 0

        # Rewards!
        self.episode_rewards = []

    def select_action(self, state):
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()  # RANDOM ACTION
            # print(f"SAMPLE ACTION TYPE: {type(action)}\n") if self.DEBUG_MODE else None
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.q_network(state)
                action = q_values.numpy()
                # print(f"Q ACTION TYPE: {type(action)}\n") if self.DEBUG_MODE else None

        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states to a tensor
        states = torch.FloatTensor(np.array(states))  # NOTE: may be silly

        actions = torch.FloatTensor(np.array(actions))
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(np.array(next_states))
        dones = torch.FloatTensor(dones)

        # Q-learning update
        q_values = self.q_network(states)
        target_q_values = self.target_network(next_states).detach()

        target = rewards + self.gamma * (1 - dones) * target_q_values.max(dim=1)[0]
        target = target.unsqueeze(1).expand_as(q_values)
        loss = self.loss_fn(
            q_values, target
        )  # changed to make match in size during loss

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.q_network_iteration == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train(self, num_episodes, save=True):
        for episode in tqdm(range(num_episodes)):
            state, _ = self.env.reset()
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = self.env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay(self.batch_size)

                total_reward += reward
                state = next_state

                if done:
                    break

            if episode % (num_episodes/10) == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")

            # Append episode reward to the list
            self.episode_rewards.append(total_reward)

        # After training, plot and save the static results
        self.plot_rewards(self.episode_rewards)

        if save == True:
            print("boutta save the model")
            self.save_model()

    def plot_rewards(self, reward_list):
        plt.figure()
        plt.plot(reward_list, label="Total Reward")
        plt.xlabel("Episodes")
        plt.ylabel("Total Reward")
        plt.title("Total Reward per Episode")
        plt.legend()
        plt.show()

    def save_model(self, model_path="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved! to {model_path}")

    def load_model(self, model_path="dqn_model.pth"):
        if os.path.exists(model_path):
            self.q_network.load_state_dict(torch.load(model_path))
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )  # Ensure consistency with target network
            print("Model loaded successfully.")
        else:
            print("Agent Model file not found.")