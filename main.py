from simple_rexy.envs.rexy_env import SimpleRexyEnv
from Agent import DQNAgent
import pybullet as p
from tqdm import tqdm


def train_rexy(num_episodes, save=True, epoch_steps=1000, total_epochs=200):
    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', BATCH_SIZE=128, LR=0.01, GAMMA=0.90,
                     EPSILON=0.9, EPSILON_DECAY=0.995, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000,
                     Q_NETWORK_ITERATION=100)

    agent.load_model(r"Epochs/dqn_model_epoch_{epoch}.pth")

    # Train the agent
    total_steps = 0
    for epoch in range(total_epochs):
        for episode in tqdm(range(num_episodes)):
            state, _ = agent.env.reset()
            total_reward = 0

            while True:
                action = agent.select_action(state)
                next_state, reward, done, _ = agent.env.step(action)
                agent.remember(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                if done:
                    break

            agent.replay(agent.batch_size)

        total_steps += (episode + 1)  # Accumulate total steps for epsilon decay

        if epoch % 10 == 0:
            print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}")

        # Decay epsilon based on total steps
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Save model at the end of each epoch
        if save:
            agent.save_model(r"Epochs/dqn_model_epoch_{epoch}.pth")

    # After training, plot and save the static results
    agent.plot_rewards(agent.episode_rewards)

if __name__ == "__main__":
    train_rexy(num_episodes=1000, save=True, epoch_steps=250, total_epochs=100)
    # run_rendered(500)  # steps
