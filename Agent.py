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
from torch.cuda.amp import autocast, GradScaler

# Define the neural network for DQN
class Net(nn.Module):
    def __init__(self, num_states, num_actions):
        super(Net, self).__init__()
        self.fc1 = nn.Linear(num_states, 256)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(128, num_actions)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu1(x)
        x = self.fc2(x)
        x = self.relu2(x)
        action_values = self.fc3(x)
        return action_values

# Define the DQN agent
class DQNAgent:
    def __init__(self, env, client, K, track_tf = False, BATCH_SIZE=128, LR=0.01, GAMMA=0.90, 
                 EPSILON=0.9, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000, 
                 Q_NETWORK_ITERATION=100):
        
        #Cuda
        if torch.cuda.is_available():
            self.device = torch.device("cuda")
            print(f"GPU {torch.cuda.get_device_name(0)} is available.")
        else:
            self.device = torch.device("cpu")
            print("GPU is not available. Using CPU.")

        # ESTABLISH REXY ENV
        self.client = client
        self.K = K
        self.env = gym.make(env, client = self.client, K = self.K).unwrapped
        self.num_actions = 2^6 #discretized
        self.num_states = self.env.observation_space.shape[0]
        self.global_step = 0
        self.episode_step = 0
        self.global_episode = 0 
        self.global_epoch = 0
        self.last_ep_avg_reward = None

        # Q-Networks
        self.LR = LR #learning rate
        self.q_network = Net(self.num_states, self.num_actions).to(self.device)
        self.target_network = copy.deepcopy(self.q_network).to(self.device)
        self.optimizer = torch.optim.Adam(self.q_network.parameters(), lr=self.LR, weight_decay= .01) #adds ridge regression 
        self.loss_fn = nn.MSELoss()

        # Experience replay
        self.MEMORY_CAPACITY = MEMORY_CAPACITY
        self.memory = deque(maxlen = self.MEMORY_CAPACITY)#.to(self.device)

        # Batch, Gamma
        self.batch_size = BATCH_SIZE
        self.gamma = GAMMA
        self.q_network_iteration = Q_NETWORK_ITERATION

        # Epsilon-greedy strategy
        self.og_epsilon = EPSILON #static
        self.epsilon = EPSILON #changes
        self.epsilon_min = EPSILON_MIN

        # Counter for Q-network updates
        self.update_counter = 0

        #Tensorboard tracking
        self.track_tf = track_tf
        self.writer = self.make_writer('jointinputs') if self.track_tf else None
        self.after_training = False
        
        #AMP COMPONENTS
        self.scaler = GradScaler()

        #Model- for directory for saving
        self.model = None
        
    def make_writer(self, log_dir):
        # Generate a timestamp
        timestamp = datetime.datetime.now().strftime('%m%d-%H%M')
        # Append the timestamp to the log directory name
        log_dir = os.path.join('TBoardFiles', log_dir, timestamp)
        if not os.path.exists(log_dir):
            os.makedirs(log_dir)
        writer = SummaryWriter(log_dir)
        return writer

    def select_action(self, state):
        """
        Chooses action based on epsilon-greedy method.
        Note - Encourages exploration to start
        """
        if self.episode_step == 1 or np.random.rand() <= self.epsilon:
            action = random.randint(0, 63)  # RANDOM Number
        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                q_values = self.q_network(state.unsqueeze(0))  # Add batch dimension
                max_q_index = torch.argmax(q_values).item()  # Get the index of the max Q-value
                action = max_q_index
        return action

    def remember(self, state, action, reward, next_state, done):
        self.memory.append((state, action, reward, next_state, done))
    #%% replay
    def replay(self, batch_size):
        if len(self.memory) < batch_size:
            return

        batch = random.sample(self.memory, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states = torch.FloatTensor(np.array(states)).to(self.device)
        actions = torch.FloatTensor(np.array(actions)).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(np.array(next_states)).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        self.q_network.to(self.device)
        self.target_network.to(self.device)

        self.optimizer.zero_grad()

        with autocast():
            q_values = self.q_network(states)
            target_q_values = self.target_network(next_states).detach() #detach prevents gradient flow
            target = rewards + self.gamma * (1 - dones) * target_q_values.max(dim=1)[0]
            target = target.unsqueeze(1).expand_as(q_values)
            loss = self.loss_fn(q_values, target)

        self.scaler.scale(loss).backward()
        self.scaler.step(self.optimizer)
        self.scaler.update()

        # Update target network and global step
        self.update_counter += 1
        if self.update_counter % self.q_network_iteration == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())

        # Logging
        if self.track_tf:
            self.writer.add_scalar('Loss/Replay', loss.item(), self.global_step)
            self.writer.add_scalar('Q-Value/Average', q_values.mean().item(), self.global_step)

        return loss
    #%% training
    def train(self, model = None, num_epochs=200, num_episodes=300, save=True, profile=False, plot=False, weights = False):
        self.model = model

        if profile:
            profiler = cProfile.Profile()
            profiler.enable()

        if weights and plot:
            self.visualize_weights(save = save)

        # Initialize a NumPy array for storing rewards
        epoch_rewards = np.zeros((num_epochs, num_episodes))
        loss_per_ep = np.zeros((num_epochs, num_episodes)) #loss per episode
        epsilon_vals = np.zeros((num_epochs, num_episodes))

        for epoch in range(num_epochs):
            self.global_epoch += 1
            for episode in tqdm(range(num_episodes)):
                self.global_episode += 1
                state, _ = self.env.reset()
                total_episode_reward = 0

                while True:
                    self.global_step += 1
                    self.episode_step += 1
                    action = self.select_action(state)
                    jvals = state[8:14] #hardcoded- joint vals
                    next_state, reward, done, newbest = self.env.step(jvals, action)
                    self.remember(state, action, reward, next_state, done) #appends to replay buffer
                    total_episode_reward += reward
                    state = next_state

                    if newbest == True and self.episode_step > 100:
                        print(f'Survived for more than 3 seconds and high score! ({total_episode_reward:.3f})')
                        self.save_model(model[:-4]+'BEST'+'.pth')

                    if done:
                        self.episode_step = 0
                        self.writer.add_scalar('Reward/Episode', total_episode_reward, self.global_episode) if self.track_tf else None
                        break

                    # Calculate the decay factor - in the episode now
                    decay_factor = (self.og_epsilon - self.epsilon_min) / 1000000 #epsilon decays at 1 million steps
                    self.epsilon = max(self.epsilon_min, self.epsilon - decay_factor)
                
                # Store the episode reward in the NumPy array
                epoch_rewards[epoch, episode] = total_episode_reward

                # Decay epsilon here, post-episode
                loss = self.replay(self.batch_size)
                loss_per_ep[epoch, episode] = loss
                epsilon_vals[epoch,episode] = self.epsilon * 100 #SCALING FOR GRAPH

            avg_reward = np.mean(epoch_rewards[epoch, :])
            print(f"Epoch {epoch+1} completed.\nAfter {episode+1} episodes, Total Avg Reward: {avg_reward:.2f}. Epsilon is {self.epsilon:.3f} ({self.global_step} steps total)")

            if avg_reward > 20000:
                print("\n\nAVERAGE REWARD > 20000! Done!\n\n")
                self.save_model(model)
                break

            if save and (num_epochs < 10 or epoch % (num_epochs // 10) == 0):
                self.save_model(model)

            #self.adjust_hypers(num_epochs, avg_reward) # novel method for epsilon decay?
                
        if profile:
            profiler.disable()
            new_prof_filepath = self.create_and_move_prof_file()
            profiler.dump_stats(new_prof_filepath)

        if plot:
            print('finna plot')
            self.visualize_weights(save = save) if weights else None
            self.plot_rewards(epoch_rewards, save = save)
            self.plot_rewards_and_epsilon(epoch_rewards, epsilon_vals, save = save)
            self.plot_loss_over_time(loss_per_ep, save = save) #removes NaNs
            input('Press enter to close the graphs.')
        self.writer.close() if self.track_tf else None

    #%% Helper Functions
    def adjust_hypers(self, num_epochs, avg_reward):
        self.epsilon = max(self.epsilon_min, self.og_epsilon * (.99 ** (4*self.global_epoch)))
        print(f'new epsilon is {self.epsilon:.3f} ({self.og_epsilon} * .99 ** (4*{self.global_epoch}))')
        #if more than a third done, and the reward is between 0 and 5% better
        """if self.global_epoch > (num_epochs // 3) and (1.05 * self.last_ep_avg_reward > avg_reward > self.last_ep_avg_reward):
            print('low rate of increase in reward! reducing learning rate')
            self.learn"""
        self.last_ep_avg_reward = avg_reward

    def save_model(self, model_path="dqn_model.pth"):
        model_path = os.path.join('.pth Files', model_path)
        torch.save(self.q_network.state_dict(), model_path)
        print(f"model saved! to {model_path}")
        #print("Q-Network Weights:")
        #for name, param in self.q_network.state_dict().items():
            #print(name, param)

    def load_model(self, model_path="dqn_model.pth"):
        model_path = os.path.join('.pth Files', model_path)
        if os.path.exists(model_path):
            self.q_network.load_state_dict(torch.load(model_path, map_location=self.device))
            self.target_network.load_state_dict(
                self.q_network.state_dict()
            )  # Ensure consistency with the target network
            print("Model loaded successfully.")
        else:
            print(f"Model file '{model_path}' not found.")

    def plot_rewards(self, epoch_rewards, save, subplot_size=(4, 3)):
        epsilon = self.og_epsilon; learning_rate= self.LR 
        gamma = self.gamma; memory_capacity = self.MEMORY_CAPACITY; batch_size = self.batch_size ; K = self.K
        num_epochs = len(epoch_rewards)
        num_epochs_to_display = min(9, num_epochs)
        epochs_to_display = np.linspace(1, num_epochs, num_epochs_to_display, dtype=int)

        rows = int(np.sqrt(num_epochs_to_display))
        cols = int(np.ceil(num_epochs_to_display / rows))

        fig, axs = plt.subplots(rows, cols, figsize=(cols * subplot_size[0], rows * subplot_size[1]))
        fig.suptitle(f"Rewards per Epoch - ε: {epsilon}, LR: {learning_rate}, γ: {gamma}, Memory: {memory_capacity}, Batch: {batch_size}, K: {K}")

        if rows * cols == 1:
            axs = [axs]  # Make axs a list if there's only one subplot

        for i, epoch_index in enumerate(epochs_to_display):
            row = i // cols
            col = i % cols
            episode_rewards = epoch_rewards[epoch_index - 1]  # Adjust for 0-based indexing

            if episode_rewards is not None and len(episode_rewards) > 0:
                ax = axs[row][col] if rows > 1 else axs[col]
                ax.plot(episode_rewards) if i != 1 else ax.plot(episode_rewards[5:])  # Skip very first one
                ax.set_title(f"Epoch {epoch_index}")
                ax.set_xlabel("Episode")
                ax.set_ylabel("Reward")
            else:
                if rows > 1:
                    axs[row][col].set_title(f"Epoch {epoch_index} (No Data)")
                    axs[row][col].axis('off')
                else:
                    axs[col].set_title(f"Epoch {epoch_index} (No Data)")
                    axs[col].axis('off')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        plt.show(block=False)
        if save:
            results_dir = self.make_results_dir()
            plot_filename = os.path.join(results_dir, 'selected_epoch_rewards.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")


    def plot_rewards_and_epsilon(self, epoch_rewards, epsilon_vals, save):
        num_epochs, num_episodes = epoch_rewards.shape

        # Flatten the epoch_rewards and epsilon_vals to a single array each
        total_rewards = epoch_rewards.flatten()
        total_epsilon_vals = epsilon_vals.flatten()

        # Create a figure and plot the rewards
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax1.set_xlabel("Total Episode")
        ax1.set_ylabel("Rewards", color='blue')
        ax1.plot(np.arange(1, num_epochs * num_episodes + 1), total_rewards, color='blue', label="Rewards")
        ax1.tick_params(axis='y', labelcolor='blue')
        ax1.legend(loc="upper left")

        # Create a secondary y-axis for epsilon values on the right side
        ax2 = ax1.twinx()
        ax2.set_ylabel("Epsilon", color='red')
        ax2.plot(np.arange(1, num_epochs * num_episodes + 1), total_epsilon_vals, color='red', linestyle='--', label="Epsilon")
        ax2.tick_params(axis='y', labelcolor='red')
        ax2.legend(loc="upper right")

        plt.title("Rewards and Epsilon Over All Episodes")
        plt.grid(True)
        plt.show(block=False)

        if save:
            results_dir = self.make_results_dir()
            plot_filename = os.path.join(results_dir, 'TotalRewardsAndEpsilon.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")

    def plot_loss_over_time(self, loss_per_ep, save, smoothing_coefficient=0.8):
        plt.figure(figsize=(8, 6))
        # Flatten the loss_per_ep array to get a 1D array for x-axis
        episode_numbers = np.arange(1, loss_per_ep.size + 1)
        # Plot the unsmoothed data in a lighter shade
        plt.plot(episode_numbers, loss_per_ep.flatten(), alpha=0.3, label="Unsmoothed Loss per Episode", color='gray')
        # Calculate smoothed loss
        smoothed_loss = np.zeros_like(loss_per_ep, dtype=float)
        for i in range(loss_per_ep.shape[0]):
            smoothed_loss[i, 0] = loss_per_ep[i, 0]
            for j in range(1, loss_per_ep.shape[1]):
                smoothed_loss[i, j] = smoothing_coefficient * smoothed_loss[i, j - 1] + (1 - smoothing_coefficient) * loss_per_ep[i, j]
        # Flatten the smoothed_loss array for y-axis focusing
        smoothed_loss_flat = smoothed_loss.flatten()
        
        # Focus the y-axis on the smoothed data
        min_smoothed_loss = np.nanmin(smoothed_loss_flat) 
        max_smoothed_loss = np.nanmax(smoothed_loss_flat) 
        plt.ylim(min_smoothed_loss, max_smoothed_loss)
        
        # Plot the smoothed loss
        plt.plot(episode_numbers, smoothed_loss_flat, label="Smoothed Loss per Episode")
        
        plt.title(f"Smoothed Loss per Episode Over Time (Smoothing: {smoothing_coefficient})")
        plt.xlabel("Episode")
        plt.ylabel("Loss")
        plt.legend()
        plt.grid(True)
        plt.show(block = False)
        if save:
            results_dir = self.make_results_dir()
            plot_filename = os.path.join(results_dir, 'AllLoss.png')
            plt.savefig(plot_filename)
            print(f"Plot saved to {plot_filename}")
    
    def visualize_weights(self, save):
        if self.after_training:
            # Get the weights of the Q-network's layers after training
            fc1_weights = self.q_network.fc1.weight.data.cpu().numpy()
            fc2_weights = self.q_network.fc2.weight.data.cpu().numpy()
            fc3_weights = self.q_network.fc3.weight.data.cpu().numpy()
            
            # Calculate the change in weights
            fc1_weights_diff = fc1_weights - self.initial_fc1_weights
            fc2_weights_diff = fc2_weights - self.initial_fc2_weights
            fc3_weights_diff = fc3_weights - self.initial_fc3_weights

            # Plot the change in weights for each layer
            fig, axs = plt.subplots(1, 3, figsize=(12, 6))

            im0 = axs[0].imshow(fc1_weights_diff, cmap='coolwarm', aspect='auto')
            axs[0].set_title('FC1 Weight Change')
            
            im1 = axs[1].imshow(fc2_weights_diff, cmap='coolwarm', aspect='auto')
            axs[1].set_title('FC2 Weight Change')
            
            im2 = axs[2].imshow(fc3_weights_diff, cmap='coolwarm', aspect='auto')
            axs[2].set_title('FC3 Weight Change')

            # Add a legend
            fig.colorbar(im0, ax=axs[0], label='Change')
            fig.colorbar(im1, ax=axs[1], label='Change')
            fig.colorbar(im2, ax=axs[2], label='Change')

            plt.tight_layout()  # Adjust layout to fit better
            plt.show(block = False)
            if save:
                results_dir = self.make_results_dir()
                plot_filename = os.path.join(results_dir, 'Weights.png')
                plt.savefig(plot_filename)
                print(f"Plot saved to {plot_filename}")
        else:
            # Log the weights before training
            self.initial_fc1_weights = self.q_network.fc1.weight.data.cpu().numpy()
            self.initial_fc2_weights = self.q_network.fc2.weight.data.cpu().numpy()
            self.initial_fc3_weights = self.q_network.fc3.weight.data.cpu().numpy()
            self.after_training = True
            pass
        
    def make_results_dir(self):
        # Create a shorter title with key parameters and NN layer sizes
        dir_name = f"{self.model}_20-256-128-6nn_eps{self.og_epsilon}_LR{self.LR}_g{self.gamma}_mem{self.MEMORY_CAPACITY}_batch{self.batch_size}_K{self.K}"
        # Construct the full path to the directory
        results_dir = os.path.join("Results Plots", dir_name)
        # Create the directory if it doesn't exist
        if not os.path.exists(results_dir):
            os.makedirs(results_dir)
        return results_dir
    
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