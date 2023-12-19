from Agent import DQNAgent
import pybullet as p
from NoLearningTester import *

def train_rexy(model=None, K = 3, num_epochs=200, num_episodes=300, save=True, track_tf = False, profile=False, plot=False, weights = False):

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', p.connect(p.DIRECT), K = K, track_tf = track_tf, BATCH_SIZE=128, LR=0.0008, GAMMA=0.99,
                     EPSILON=1, EPSILON_MIN=0.1, MEMORY_CAPACITY=900000,
                     Q_NETWORK_ITERATION=100)
    agent.load_model(model)

    agent.train(model=model, num_epochs=num_epochs, 
                num_episodes=num_episodes, save=save, 
                profile=profile, plot=plot, weights = weights)
    
if __name__ == "__main__":
    
    train_rexy(model='juststandup4.pth', K = 2,
            num_epochs=500, num_episodes=1000,
            track_tf = 0, save=1, plot = 1, profile = 0,
            weights = True)
        
    #run_trained_agent('juststandupKis4.pth', K = 4)