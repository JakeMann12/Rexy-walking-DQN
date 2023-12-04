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

#ESTABLISH REXY ENV
env = gym.make('rexy-v0')
client = env.client
env = env.unwrapped
NUM_ACTIONS = env.action_space.shape[0]
NUM_STATES = env.observation_space.shape[0]

# Hyperparameters
BATCH_SIZE = 128
LR = 0.01 #learning rate
GAMMA = 0.90
EPSILON = 0.9
EPSILON_DECAY = .995
EPSILON_MIN = .01
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100 #how many steps before updating the network

#DEBUGGING TOGGLE
debug = True

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
    def __init__(self, num_states, num_actions):
        self.num_states = num_states
        self.num_actions = num_actions

        # Q-Networks
        self.q_network = Net(num_states, num_actions)
        self.target_network = copy.deepcopy(self.q_network)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=LR)
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.memory = deque(maxlen=MEMORY_CAPACITY)

        # Epsilon-greedy strategy
        self.epsilon = EPSILON
        self.epsilon_decay = EPSILON_DECAY
        self.epsilon_min = EPSILON_MIN

        # Counter for Q-network updates
        self.update_counter = 0

        #Rewards!
        self.episode_rewards = []

        self.DEBUG_MODE = debug

    def select_action(self, state):
        if np.random.rand() <= EPSILON:
            action = env.action_space.sample() #RANDOM ACTION
            #print(f"SAMPLE ACTION TYPE: {type(action)}\n") if self.DEBUG_MODE else None
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state)
                q_values = self.q_network(state)
                action = q_values.numpy()
                #print(f"Q ACTION TYPE: {type(action)}\n") if self.DEBUG_MODE else None

        return action
    
    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))

    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        # Convert states to a tensor
        states = torch.FloatTensor(states) #NOTE: may be silly

        actions = torch.FloatTensor(actions)
        rewards = torch.FloatTensor(rewards)
        next_states = torch.FloatTensor(next_states)
        dones = torch.FloatTensor(dones)

        # Q-learning update
        q_values = self.q_network(states)
        target_q_values = self.target_network(next_states).detach()

        target = rewards + GAMMA * (1 - dones) * target_q_values.max(dim=1)[0]
        loss = self.loss_fn(q_values, target.unsqueeze(1))

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % Q_NETWORK_ITERATION == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Decay epsilon
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)


    def train(self, num_episodes):

        for episode in tqdm(range(num_episodes)):
            state, _ = env.reset()
            total_reward = 0

            while True:
                action = self.select_action(state)
                next_state, reward, done, _ = env.step(action)
                self.remember(state, action, reward, next_state, done)
                self.replay(BATCH_SIZE)

                total_reward += reward
                state = next_state

                if done:
                    break

            if episode % 5 == 0:
                print(f"Episode {episode}, Total Reward: {total_reward}")
            
            # Append episode reward to the list
            self.episode_rewards.append(total_reward)

        # After training, plot and save the static results
        self.plot_rewards(self.episode_rewards)
        print('boutta save the model')
        self.save_model()

    def plot_rewards(self, reward_list):
        plt.figure()
        plt.plot(reward_list, label='Total Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.show()

    def save_model(self, model_path="dqn_model.pth"):
        torch.save(self.q_network.state_dict(), model_path)

    def load_model(self, model_path="dqn_model.pth"):
        if os.path.exists(model_path):
            self.q_network.load_state_dict(torch.load(model_path))
            self.target_network.load_state_dict(self.q_network.state_dict())  # Ensure consistency with target network
            print("Model loaded successfully.")
        else:
            print("Model file not found.")


if __name__=='__main__':

    # Initialize DQNAgent
    agent = DQNAgent(NUM_STATES, NUM_ACTIONS)

    # Train the agent
    agent.train(num_episodes=200)
