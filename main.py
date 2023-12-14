from Agent import DQNAgent
import matplotlib.pyplot as plt
import numpy as np
import cProfile
import shutil
import os
import datetime

def create_and_move_prof_file():
    # Create a folder for profiler outputs if it doesn't exist
    profiler_output_folder = "ProfilerOutputs"
    if not os.path.exists(profiler_output_folder):
        os.makedirs(profiler_output_folder)

    # Get the current date and time in the desired format (e.g., 'Dec13-14:58')
    current_time = datetime.datetime.now().strftime('%b%d-%H%M')
    
    # Move existing .prof files to ProfilerOutputs folder
    for filename in os.listdir():
        if filename.endswith(".prof"):
            shutil.move(filename, os.path.join(profiler_output_folder, filename))

    # Create a new .prof file with the current date/time
    new_prof_filename = f"{current_time}.prof"
    new_prof_filepath = os.path.join(new_prof_filename)

    # Return the path to the newly created .prof file
    return new_prof_filepath

def train_rexy(model=None, num_epochs=200, num_episodes=300, save=True, profile=False, plot=False):
    if profile:
        profiler = cProfile.Profile()
        profiler.enable()

    epoch_rewards = []  # List to store rewards for each epoch

    # Initialize DQNAgent
    agent = DQNAgent('rexy-v0', BATCH_SIZE=256, LR=0.2, GAMMA=0.90,
                     EPSILON=0.9, EPSILON_DECAY=0.9975, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000,
                     Q_NETWORK_ITERATION=100)
    agent.load_model(model)

    # Train the agent
    for epoch in range(num_epochs):
        # Call the train method of the agent, which now returns episode_rewards
        episode_rewards = agent.train(num_episodes)

        # Append the episode_rewards to epoch_rewards
        epoch_rewards.append(episode_rewards)

        # Add code to evaluate performance, adjust hyperparameters,
        # save the model, or perform any other epoch-level operations in the future.
        print(f"Epoch {epoch + 1} Completed")

    if save:
        agent.save_model(model)

    if profile:
        profiler.disable()
        new_prof_filepath = create_and_move_prof_file()
        profiler.dump_stats(new_prof_filepath)

    if plot:
        plot_rewards(epoch_rewards)

def plot_rewards(epoch_rewards, subplot_size=(4, 3)):
    num_epochs = len(epoch_rewards)
    num_epochs_to_display = min(9, num_epochs)
    epochs_to_display = np.linspace(1, num_epochs, num_epochs_to_display, dtype=int)

    rows = int(np.sqrt(num_epochs_to_display))
    cols = int(np.ceil(num_epochs_to_display / rows))

    fig, axs = plt.subplots(rows, cols, figsize=(cols * subplot_size[0], rows * subplot_size[1]))
    fig.suptitle("Rewards per Epoch")

    for i, epoch_index in enumerate(epochs_to_display):
        row = i // cols
        col = i % cols
        episode_rewards = epoch_rewards[epoch_index - 1]  # Adjust for 0-based indexing

        if episode_rewards is not None and len(episode_rewards) > 0:
            axs[row, col].plot(episode_rewards)
            axs[row, col].set_title(f"Epoch {epoch_index}")
            axs[row, col].set_xlabel("Episode")
            axs[row, col].set_ylabel("Reward")
        else:
            axs[row, col].set_title(f"Epoch {epoch_index} (No Data)")
            axs[row, col].axis('off')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # Show the plot and block the code until the user closes the plot window
    plt.show()

if __name__ == "__main__":
    train_rexy(model='dqn_model.pth', 
               num_epochs=20, num_episodes=400, 
               save=False, plot = True, profile=False)
