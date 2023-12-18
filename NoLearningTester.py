import gym
import torch

# from agent import TRPOAgent
from simple_rexy.envs.rexy_env import SimpleRexyEnv
import time
import pybullet as p
import matplotlib.pyplot as plt
import numpy as np
from Agent import DQNAgent
# zswang666 and electricelephant

def plot_results(rewards, xy_positions, episode):
    # Plot the reward function
    plt.figure(figsize=(14, 5))

    plt.subplot(1, 2, 1)
    plt.plot(rewards, label="Reward")
    plt.xlabel("Step")
    plt.ylabel("Reward")
    plt.title("Reward Function Over Time")
    plt.legend()

    # Plot the x-y positions
    xy_positions = list(zip(*xy_positions))
    plt.subplot(1, 2, 2)
    colors = np.linspace(
        0, 1, len(xy_positions[0])
    )  # Color based on time, green to purple
    plt.scatter(
        xy_positions[0],
        xy_positions[1],
        c=colors,
        cmap="viridis",
        label="Path",
        marker="o",
    )
    plt.scatter(
        [xy_positions[0][0]],
        [xy_positions[1][0]],
        color="green",
        marker="*",
        s=260,
        label="Start",
    )
    plt.scatter(
        [xy_positions[0][-1]],
        [xy_positions[1][-1]],
        color="red",
        marker="x",
        s=260,
        label="End",
    )
    plt.xlabel("X Position")
    plt.ylabel("Y Position")
    plt.title(f"Best Run - Reward and Path of Rexy Over Time (Ep. {episode})")
    plt.legend()
    plt.show()

def run_trained_agent(model_path, K = 2, num_episodes = 100):
    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', p.connect(p.GUI), K = K, BATCH_SIZE=32, LR=0.01, GAMMA=0.90,
                     EPSILON=0.0,  # Set epsilon to 0 for a deterministic policy
                     EPSILON_DECAY=1.0,  # No epsilon decay during the run
                     EPSILON_MIN=0.0,  # Minimum epsilon during the run
                     MEMORY_CAPACITY=1000000,
                     Q_NETWORK_ITERATION=50)

    # Load the trained model
    agent.load_model(model_path)

    # Run the trained agent without training
    best_reward = 0
    best_rewards = []
    best_xy_positions = []
    best_episode = None

    for episode in range(1,num_episodes+1): #ten episodes

        state, _ = agent.env.reset()
        ep_reward = 0
        ep_rewards = []
        ep_xy_positions = []

        while True:
            action = agent.select_action(state)
            jvals = state[8:14]
            next_state, reward, done, _ = agent.env.step(jvals, action)
            #print(next_state[-6:])
            ep_reward += reward
            ep_rewards.append(ep_reward)
            ep_xy_positions.append((next_state[0].item(), next_state[1].item()))
            state = next_state

            if done:
                if ep_reward > best_reward:
                    best_rewards = ep_rewards
                    best_xy_positions = ep_xy_positions
                    best_episode = episode
                break
    p.disconnect()
    # Plot the results
    plot_results(best_rewards, best_xy_positions, best_episode)

if __name__ == "__main__":
    run_trained_agent(r"juststandup4.pth", K = 3, num_episodes = 100)

    