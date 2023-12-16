from simple_rexy.envs.rexy_env import SimpleRexyEnv
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
import cProfile
import shutil
import os
import datetime
from torch.utils.tensorboard import SummaryWriter

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
    def __init__(self, env, client, BATCH_SIZE=128, LR=0.01, GAMMA=0.90, 
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
        self.client = client
        self.env = gym.make(env, client = self.client).unwrapped
        self.num_actions = self.env.action_space.shape[0]
        self.num_states = self.env.observation_space.shape[0]
        self.global_step = 0
        self.global_episode = 0 

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

        #Tensorboard tracking
        self.writer = self.make_writer('jointinputs')
        
    def make_writer(self, log_dir):
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        # Append the timestamp to the log directory name
        log_dir = os.path.join('TBoard Files', log_dir, timestamp)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        return writer

    def select_action(self, state):
        """
        Chooses action based on epsilon-greedy method, and increments global_step
        """
        if np.random.rand() <= self.epsilon:
            action = self.env.action_space.sample()  # RANDOM ACTION
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state)
                action = q_values.cpu().numpy()
        self.global_step += 1
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

        # Log the loss and average Q-value
        self.writer.add_scalar('Loss/Replay', loss.item(), self.global_step)
        self.writer.add_scalar('Q-Value/Average', q_values.mean().item(), self.global_step)

        #decay epsilon- back in the replay
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Update target network
        self.update_counter += 1
        if self.update_counter % self.q_network_iteration == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

    def train(self, model = None, num_epochs=200, num_episodes=300, save=True, profile=False, plot=False):
        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        epoch_rewards = []  # List to store rewards for each epoch
        for epoch in range(num_epochs):

            episode_rewards = []
            for episode in tqdm(range(num_episodes)):
                state, _ = self.env.reset()
                total_episode_reward = 0

                while True:
                    self.global_episode += 1
                    action = self.select_action(state)
                    next_state, reward, done, newbest = self.env.step(action)
                    self.remember(state, action, reward, next_state, done)
                    total_episode_reward += reward
                    state = next_state

                    if newbest == True and self.current_step > 100:
                        print('got a new best reward for this run!')
                        self.save_model(model+'BEST')
                    # NOTE: moved Decay epsilon into each step instead of in the replay funct.
                    if done:
                        self.writer.add_scalar('Reward/Episode', total_episode_reward, self.global_episode)
                        break

                #NOTE: put this back in every episode if it takes too long time-wise to learn.
                self.replay(self.batch_size)

                # Append episode reward to the list
                episode_rewards.append(total_episode_reward)

                if episode == num_episodes - 1:
                    print(f"After {episode+1} episodes, Total Avg Reward: {np.mean(episode_rewards):.2f}")

            epoch_rewards.append(episode_rewards)
            print(f"Epoch {epoch + 1} Completed")

            if save and epoch % (num_epochs // 10) == 0:
                self.save_model(model)

        if profile:
            profiler.disable()
            new_prof_filepath = self.create_and_move_prof_file()
            profiler.dump_stats(new_prof_filepath)

        if plot:
            self.plot_rewards(epoch_rewards)

        self.writer.close()

    def save_model(self, model_path="dqn_model.pth"):
        model_path = os.path.join('.pth Files', model_path)
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved! to {model_path}")

    def load_model(self, model_path="dqn_model.pth"):
        model_path = os.path.join('.pth Files', model_path)
        if os.path.exists(model_path):
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )  # Ensure consistency with target network
            print("Model loaded successfully.")
        else:
            print("Agent Model file not found.")

    def plot_rewards(self, epoch_rewards, subplot_size=(4, 3)):
        num_epochs = len(epoch_rewards)
        num_epochs_to_display = min(9, num_epochs)
        epochs_to_display = np.linspace(1, num_epochs, num_epochs_to_display, dtype=int)

        rows = int(np.sqrt(num_epochs_to_display))
        cols = int(np.ceil(num_epochs_to_display / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * subplot_size[0], rows * subplot_size[1]))
        fig.suptitle("Rewards per Epoch")

        for i, epoch_index in enumerate(epochs_to_display):
            row = i // cols
            col = i % cols
            episode_rewards = epoch_rewards[epoch_index - 1]  # Adjust for 0-based indexing

            if episode_rewards is not None and len(episode_rewards) > 0:
                if num_epochs_to_display > 1:
                    ax = axs[row, col]
                else:
                    ax = axs  # When displaying a single plot, axs is not a 2D array

                ax.plot(episode_rewards) if i != 1 else ax.plot(episode_rewards[5:]) #skip very first one
                ax.set_title(f"Epoch {epoch_index}")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
            else:
                axs[row, col].set_title(f"Epoch {epoch_index} (No Data)")
                axs[row, col].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Show the plot and block the code until the user closes the plot window
        plt.show()

    def create_and_move_prof_file(self):
        # Specify the ProfilerOutputs directory
        profiler_output_folder = "ProfilerOutputs"
        # Clear the contents of the ProfilerOutputs directory (including subdirectories)
        if os.path.exists(profiler_output_folder):
            shutil.rmtree(profiler_output_folder)
        # Recreate the empty ProfilerOutputs directory
        os.makedirs(profiler_output_folder)
        # Get the current date and time in the desired format (e.g., 'Dec13-14:58')
        current_time = datetime.datetime.now().strftime('%b%d-%H%M')
        # Move existing .prof files to ProfilerOutputs folder
        for filename in os.listdir():
            if filename.endswith(".prof"):
                shutil.move(filename, os.path.join(profiler_output_folder, filename))
        # Create a new .prof file with the current date/time
        new_prof_filename = f"{current_time}.prof"
        new_prof_filepath = os.path.join(profiler_output_folder, new_prof_filename)
        # Return the path to the newly created .prof file
        return new_prof_filepath