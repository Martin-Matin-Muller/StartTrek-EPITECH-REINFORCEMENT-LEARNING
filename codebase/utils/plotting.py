import matplotlib.pyplot as plt
import numpy as np
import os

# Plot Function ===========================================================================
def save_graph(self, rewards_per_episode, epsilon_history):
    # Save plots
    fig = plt.figure(1)

    # Plot average rewards (Y-axis) vs episodes (X-axis)
    mean_rewards = np.zeros(len(rewards_per_episode))
    for x in range(len(mean_rewards)):
        mean_rewards[x] = np.mean(rewards_per_episode[max(0, x-99):(x+1)])
    plt.subplot(121) # plot on a 1 row x 2 col grid, at cell 1
    # plt.xlabel('Episodes')
    plt.ylabel('Mean Rewards')
    plt.plot(mean_rewards)

    # Plot epsilon decay (Y-axis) vs episodes (X-axis)
    plt.subplot(122) # plot on a 1 row x 2 col grid, at cell 2
    # plt.xlabel('Time Steps')
    plt.ylabel('Epsilon Decay')
    plt.plot(epsilon_history)

    plt.subplots_adjust(wspace=1.0, hspace=1.0)

    # Save plots
    fig.savefig(self.GRAPH_FILE)
    plt.close(fig)

def plot_scores(scores, save_path="scores.png", window=100):

    if len(scores) == 0:
        print("Aucun score à afficher")
        return

    os.makedirs(os.path.dirname(save_path), exist_ok=True) if os.path.dirname(save_path) else None

    plt.figure()

    x = np.arange(len(scores))

    # Courbe des scores
    plt.plot(x, scores, label="Scores")

    # Moyenne glissante
    if len(scores) >= window:
        moving_avg = np.convolve(scores, np.ones(window)/window, mode='valid')
        plt.plot(np.arange(len(moving_avg)), moving_avg, color='red',
                 label=f"Moyenne glissante ({window})")
    else:
        print(f"Pas assez de données pour une fenêtre de {window}")

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("Training Scores")
    plt.legend()

    plt.savefig(save_path)
    print(f"Graph sauvegardé ici : {save_path}")

    plt.close()
#==========================================================================================