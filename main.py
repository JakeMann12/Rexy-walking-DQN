from simple_rexy.envs.rexy_env import SimpleRexyEnv
from SimpleAgent import DQNAgent
import pybullet as p

def train_rexy(num_episodes, save = True):
    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', BATCH_SIZE = 128, LR = 0.01, GAMMA = 0.90, 
                 EPSILON = 0.9, EPSILON_DECAY = 0.995, EPSILON_MIN = 0.01, MEMORY_CAPACITY = 2000, 
                 Q_NETWORK_ITERATION = 100)

    agent.load_model("dqn_model.pth")
    # Train the agent
    agent.train(num_episodes=num_episodes, save=save)  # not saving yet

def run_rendered(max_steps):
    agent = DQNAgent('rexy-v0', EPSILON = 0) #epsilon -> 0 only exploit
    agent.load_model("dqn_model.pth")

    state, _ = agent.env.reset()
    # Run the agent in the environment
    for _ in range(max_steps): 
        # Select an action based on the learned policy
        action = agent.select_action(state)
        next_state, reward, done, _ = agent.env.step(action)
        state = next_state
        agent.env.render()
        if done:
            break
    agent.env.close()


if __name__ == "__main__":
    train_rexy(50, save=True)
    #run_rendered(500) #steps