import gym
import torch
#from agent import TRPOAgent
from simple_rexy.envs.rexy_env import SimpleRexyEnv
import time
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np

#zswang666 and electricelephant

def plot_results(rewards, xy_positions):
    # Plot the reward function
    plt.figure(figsize=(14, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(rewards, label='Reward')
    plt.xlabel('Step')
    plt.ylabel('Reward')
    plt.title('Reward Function Over Time')
    plt.legend()

    # Plot the x-y positions
    xy_positions = list(zip(*xy_positions))
    plt.subplot(1, 2, 2)
    colors = np.linspace(0, 1, len(xy_positions[0]))  # Color based on time, green to purple
    plt.scatter(xy_positions[0], xy_positions[1], c=colors, cmap='viridis', label='Path', marker='o')
    plt.scatter([xy_positions[0][0]], [xy_positions[1][0]], color='green', marker='*', s = 260, label='Start')
    plt.scatter([xy_positions[0][-1]], [xy_positions[1][-1]], color='red', marker='x', s = 260, label='End')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.title('Path of Rexy Over Time')
    plt.legend()

    plt.show()

def main():
    env = gym.make('rexy-v0')
    observation = env.reset()

    # Lists to store data for plotting
    rewards = []
    xy_positions = []

    steps = 500
    for step in range(steps):
        action = env.action_space.sample()
        #action = np.zeros(6) #for testing
        observation, reward, done, info = env.step(action)

        rewards.append(reward)
        xy_positions.append((observation[0].item(), observation[1].item()))

        if step == 1:  # Print observation for comparison
            #If cos(Z) is close to 1, the object is mostly aligned with the X-axis. If sin(Z) is close to 1, the object is mostly aligned with the Y-axis.
            print(f"Rexy Observation (X Y Z R P Y and X Y velocity):\n{env.rexy.get_observation()}\n")

        if not env.observation_space.contains(observation):
            print("Invalid observation:", observation)

        if done:
            print(f"Episode finished after {step + 1} timesteps\nFinal Observation:\n{observation}")
            plot_results(rewards, xy_positions)
            break

        if step == steps - 1:
            plot_results(rewards, xy_positions)

    env.close()

if __name__ == '__main__':
    main()
