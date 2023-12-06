from simple_rexy.envs.rexy_env import SimpleRexyEnv
from Agent import DQNAgent
import pybullet as p
from tqdm import tqdm

def train_rexy(model, num_episodes = 300, save=True, epoch_steps=1000, total_epochs=200):
    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', BATCH_SIZE=512, LR=0.04, GAMMA=0.90,
                     EPSILON=0.9, EPSILON_DECAY=0.995, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000,
                     Q_NETWORK_ITERATION=100)

    agent.load_model(model)

    # Train the agent
    total_steps = 0
    for epoch in range(total_epochs+1):
        for episode in tqdm(range(num_episodes+1)):
            state, _ = agent.env.reset()
            total_reward = 0
            steps_in_epoch = 0

            while steps_in_epoch < epoch_steps:
                action = agent.select_action(state)
                next_state, reward, done, _ = agent.env.step(action)
                agent.remember(state, action, reward, next_state, done)

                total_reward += reward
                state = next_state

                steps_in_epoch += 1
                total_steps += 1

                if done:
                    break

            agent.replay(agent.batch_size)

        #if epoch % (total_epochs // 10) == 0:
        print(f"Epoch {epoch}, Total Reward: {total_reward:.2f}")

        # Decay epsilon based on total steps
        agent.epsilon = max(agent.epsilon_min, agent.epsilon * agent.epsilon_decay)

        # Save model at the end of each epoch
        if save:
            agent.save_model(model)

    # After training, plot and save the static results
    agent.plot_rewards(agent.episode_rewards)

if __name__ == "__main__":
    train_rexy(r"Epochs/dqn_model_epoch_{epoch}.pth", num_episodes=1000, save=True, epoch_steps=1000, total_epochs=2)
    # run_rendered(500)  # steps
