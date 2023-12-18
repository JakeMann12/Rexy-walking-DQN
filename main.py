from Agent import DQNAgent
import pybullet as p

def train_rexy(model=None, K = 3, num_epochs=200, num_episodes=300, save=True, track_tf = False, profile=False, plot=False, weights = False):

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', p.connect(p.DIRECT), K = K, track_tf = track_tf, BATCH_SIZE=128, LR=0.00075, GAMMA=0.93,
                     EPSILON=1, EPSILON_DECAY=0.99, EPSILON_MIN=0.1, MEMORY_CAPACITY=1000000,
                     Q_NETWORK_ITERATION=75)
    agent.load_model(model)

    agent.train(model=model, num_epochs=num_epochs, 
                num_episodes=num_episodes, save=save, 
                profile=profile, plot=plot, weights = weights)
    
if __name__ == "__main__":
    
    train_rexy(model='juststandup5.pth', K = 3,
            num_epochs=2, num_episodes=1000,
            track_tf = 0, save=1, plot = 1, profile = 0,
            weights = True)
        