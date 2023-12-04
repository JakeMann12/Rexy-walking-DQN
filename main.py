from simple_rexy.envs.rexy_env import SimpleRexyEnv
from SimpleAgent import DQNAgent
import pybullet as p

if __name__ == "__main__":
    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', render=True, debug=True, BATCH_SIZE = 128, LR = 0.01, GAMMA = 0.90, 
                 EPSILON = 0.9, EPSILON_DECAY = 0.995, EPSILON_MIN = 0.01, MEMORY_CAPACITY = 2000, 
                 Q_NETWORK_ITERATION = 100)

    agent.load_model("agent.pth")
    # Train the agent
    agent.train(num_episodes=200, save=False)  # not saving yet