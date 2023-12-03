import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import os
import gym
import copy
from simple_rexy.resources.REXY import Rexy
from tqdm import tqdm

#ESTABLISH REXY ENV
env = gym.make('rexy-v0')
client = env.client
env = env.unwrapped
NUM_ACTIONS = env.action_space.shape[0]
NUM_STATES = env.observation_space.shape[0]

#ESTABLISH REXY
REXY = Rexy(client)

# Hyperparameters
BATCH_SIZE = 128
LR = 0.01
GAMMA = 0.90
EPSILON = 0.9
MEMORY_CAPACITY = 2000
Q_NETWORK_ITERATION = 100 #how many steps before updating the network

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

class DQNAgent():
    """docstring for DQN"""
    def __init__(self, debug = False):
        self.eval_net, self.target_net = Net(NUM_STATES, NUM_ACTIONS), Net(NUM_STATES, NUM_ACTIONS)

        self.num_actions = NUM_ACTIONS
        self.learn_step_counter = 0
        self.memory_counter = 0
        self.memory = deque(maxlen=MEMORY_CAPACITY)
        self.optimizer = torch.optim.Adam(self.eval_net.parameters(), lr=LR)
        self.loss_func = nn.MSELoss()
        
        self.DEBUG_MODE = debug

        #self.env_instance = SimpleRexyEnv()


    def choose_action(self, state):
        state = torch.FloatTensor(state)
        print(f"\n\nState: {state}\n\n") if self.DEBUG_MODE else None
        q_values = self.eval_net(state)
        
        # Explore (with probability EPSILON) or exploit
        if np.random.rand() <= EPSILON:
            action = env.action_space.sample() #RANDOM ACTION
        else:
            action = q_values.argmax().item()
        
        return action

    def store_transition(self, state, action, reward, next_state):
        transition = np.concatenate((state, [action, reward], next_state))
        if len(self.memory) < MEMORY_CAPACITY:
            self.memory.append(transition)
        else:
            index = self.memory_counter % MEMORY_CAPACITY
            self.memory[index] = transition
        self.memory_counter += 1

    def learn(self):

        #update the parameters
        if self.learn_step_counter % Q_NETWORK_ITERATION == 0:
            self.target_net.load_state_dict(self.eval_net.state_dict())
        self.learn_step_counter += 1

        if len(self.memory) < BATCH_SIZE: #if too big
            return

        sample_index = np.random.choice(MEMORY_CAPACITY, BATCH_SIZE)
        batch_memory = self.memory[sample_index, :]
        batch_state = torch.FloatTensor(batch_memory[:, :NUM_STATES])
        batch_action = torch.LongTensor(batch_memory[:, NUM_STATES:NUM_STATES + 1].astype(int))
        batch_reward = torch.FloatTensor(batch_memory[:, NUM_STATES + 1:NUM_STATES + 2])
        batch_next_state = torch.FloatTensor(batch_memory[:, -NUM_STATES:])

        q_eval = self.eval_net(batch_state).gather(1, batch_action)
        q_next = self.target_net(batch_next_state).detach()
        q_target = batch_reward + GAMMA * q_next.max(1)[0].view(BATCH_SIZE, 1)
        loss = self.loss_func(q_eval, q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def plot_rewards(self, reward_list):
        plt.figure()
        plt.plot(reward_list, label='Total Reward')
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.legend()
        plt.show()

    def save_model(self, model_path="dqn_model.pth"):
        torch.save(self.eval_net.state_dict(), model_path)

    def load_model(self, model_path="dqn_model.pth"):
        if os.path.exists(model_path):
            self.eval_net.load_state_dict(torch.load(model_path))
            print("Model loaded successfully.")
        else:
            print("Model file not found.")

def main():
    dqn = DQNAgent(debug = True)
    episodes = 100

    # Training
    print("Collecting Experience....")
    reward_list = []

    for i in tqdm(range(episodes), desc="Processing episodes"):
        state, _ = env.reset()
        ep_reward = 0

        while True:
            # env.render()
            action = dqn.choose_action(state)
            next_state, reward, done, _ = env.step(action)
            ep_reward += reward
            dqn.store_transition(state, action, reward, next_state)
            
            if done:
                print(f"episode: {i} , the episode reward is {reward:.3f}")
                break

            state = next_state

        r = copy.copy(ep_reward)
        reward_list.append(r)

        if i % 50 == 0:
            print(f"Episode: {i}, Total Reward: {ep_reward}")

        dqn.learn()

    # Save the trained model
    dqn.save_model()

    # Load the trained model
    dqn.load_model()

    # Use the trained model for decision-making
    state = env.reset()
    for _ in range(100):  # Run for 100 steps using the trained model
        action = dqn.choose_action(state)
        next_state, _, done, _ = env.step(action)
        if done:
            break
        state = next_state

    # Plot the rewards
    dqn.plot_rewards(reward_list)

    env.close()

if __name__ == '__main__':
    main()