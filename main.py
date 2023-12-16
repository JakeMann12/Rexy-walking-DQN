from Agent import DQNAgent
import pybullet as p

def train_rexy(model=None, num_epochs=200, num_episodes=300, save=True, track_tf = False, profile=False, plot=False):

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', p.connect(p.DIRECT), track_tf = track_tf, BATCH_SIZE=256, LR=0.0006, GAMMA=0.90,
                     EPSILON=0.9, EPSILON_DECAY=0.995, EPSILON_MIN=0.05, MEMORY_CAPACITY=2000,
                     Q_NETWORK_ITERATION=100)
    agent.load_model(model)

    agent.train(model=model, num_epochs=num_epochs, 
                num_episodes=num_episodes, save=save, 
                profile=profile, plot=plot)


if __name__ == "__main__":
    
    train_rexy(model='juststandup.pth',
               num_epochs=50 , num_episodes=100, 
               track_tf = 0, save=1, plot = 1, profile = 1)
    