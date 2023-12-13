from Agent import DQNAgent
import matplotlib.pyplot as plt
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
    agent = DQNAgent('rexy-v0', BATCH_SIZE=512, LR=0.04, GAMMA=0.90,
                     EPSILON=0.9, EPSILON_DECAY=0.995, EPSILON_MIN=0.01, MEMORY_CAPACITY=2000,
                     Q_NETWORK_ITERATION=100)
    agent.load_model(model)

    # Train the agent
    for epoch in range(num_epochs):
        # Call the train method of the agent, which now returns episode_rewards
        episode_rewards = agent.train(num_episodes, save=save)

        # Append the episode_rewards to epoch_rewards
        epoch_rewards.append(episode_rewards)

        # Add code to evaluate performance, adjust hyperparameters,
        # save the model, or perform any other epoch-level operations in the future.
        print(f"Epoch {epoch + 1} Completed")

    if profile:
        profiler.disable()
        new_prof_filepath = create_and_move_prof_file()
        profiler.dump_stats(new_prof_filepath)

    if plot:
        plot_rewards(epoch_rewards)

def plot_rewards(epoch_rewards):
    # Create one big figure with subplots for each epoch
    num_epochs = len(epoch_rewards)
    fig, axs = plt.subplots(1, num_epochs, figsize=(num_epochs * 4, 3))
    fig.suptitle("Rewards per Epoch")

    for i, episode_rewards in enumerate(epoch_rewards):
        if episode_rewards is not None and len(episode_rewards) > 0:
            axs[i].plot(episode_rewards)
            axs[i].set_title(f"Epoch {i + 1}")
            axs[i].set_xlabel("Episode")
            axs[i].set_ylabel("Reward")
        else:
            axs[i].set_title(f"Epoch {i + 1} (No Data)")
            axs[i].axis('off')

    # Save the big figure
    plt.tight_layout()
    #plt.savefig("rewards_per_epoch.png")
    plt.show()

if __name__ == "__main__":
    train_rexy(model=r'Epochs/3layer128node', num_epochs=2, num_episodes=300, save=False, plot = True, profile=True)
