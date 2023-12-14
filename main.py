from Agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import cProfile
import shutil
import os
import datetime

def train_rexy(model=None, num_epochs=200, num_episodes=300, save=True, profile=False, plot=False):

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', BATCH_SIZE=512, LR=0.25, GAMMA=0.90,
                     EPSILON=0.95, EPSILON_DECAY=0.9975, EPSILON_MIN=0.05, MEMORY_CAPACITY=2250,
                     Q_NETWORK_ITERATION=80)
    agent.load_model(model)

    agent.train(model=model, num_epochs=num_epochs, num_episodes=num_episodes, save=save, profile=profile, plot=plot)


if __name__ == "__main__":
    train_rexy(model='dqn_model.pth', 
               num_epochs=15, num_episodes=100, 
               save=False, plot = True, profile = False)
