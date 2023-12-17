from Agent import DQNAgent
import pybullet as p

def train_rexy(model=None, num_epochs=200, num_episodes=300, save=True, track_tf = False, profile=False, plot=False, weights = False):

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', p.connect(p.DIRECT), track_tf = track_tf, BATCH_SIZE=128, LR=0.00075, GAMMA=0.93,
                     EPSILON=1.0, EPSILON_DECAY=0.99, EPSILON_MIN=0.1, MEMORY_CAPACITY=15000,
                     Q_NETWORK_ITERATION=100)
    agent.load_model(model)

    agent.train(model=model, num_epochs=num_epochs, 
                num_episodes=num_episodes, save=save, 
                profile=profile, plot=plot, weights = weights)
    

if __name__ == "__main__":
    
    try:
        train_rexy(model='juststandup.pth',
                num_epochs=3000, num_episodes=100, 
                track_tf = 0, save=1, plot = 0, profile = 0, 
                weights = True)
    except:
        train_rexy(model='juststandup.pth',
                num_epochs=3000, num_episodes=1000, 
                track_tf = 0, save=1, plot = 0, profile = 0, 
                weights = True)