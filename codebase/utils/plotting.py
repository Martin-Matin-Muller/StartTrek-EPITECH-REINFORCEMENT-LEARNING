import matplotlib.pyplot as plt
import numpy as np
import os
from datetime import datetime
from ..log_variable import DATE_FORMAT 

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
        print("No score to print")
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
                 label=f"Average reward ({window})")
    else:
        print(f"Not enough data for window of {window}")

    plt.ylabel('Score')
    plt.xlabel('Episode #')
    plt.title("Training Scores")
    plt.legend()

    plt.savefig(save_path)
    print(f"Graph saved here : {save_path}")

    plt.close()

def classify_termination(self, state, terminated, truncated):
    x, y, vx, vy, angle, _, left, right = state

    if truncated:
        return "time_limit"

    if terminated:
        # Successful landing
        if left and right and abs(vx) < 0.1 and abs(vy) < 0.1 and abs(angle) < 0.1:
            return "landed"

        # Out of bounds
        if abs(x) > 1.0:
            return "out_of_bounds"

        # Crash - too fast
        if abs(vy) > 0.5:
            return "crash_high_speed"

        # Crash - bad angle
        if abs(angle) > 0.4:
            return "crash_bad_angle"

    return "unknown"

def update_termination_stats(self, state, terminated, truncated):
    reason = self.classify_termination(state, terminated, truncated)
    self.termination_stats[reason] += 1

def log_with_termination_stats(self, log_file, episode, last_100_episode_mean, eps, last_graph_update_time, start_time):
    total = sum(self.termination_stats.values())

    header = "=" * 50
    title = f"Termination stats at episode {episode}"

    lines = []
    lines.append(header)
    lines.append(title)
    lines.append(header)

    if total == 0:
        lines.append("No data yet.")
    else:
        for key, value in self.termination_stats.items():
            percent = (value / total) * 100 if total > 0 else 0
            lines.append(f"{key:<20} : {value:5d} ({percent:6.2f}%)")

    lines.append(header)

    elapsed = datetime.now() - start_time
    elapsed_str = str(elapsed).split('.')[0]  # removes microseconds


    
    log_message = (
        f"{elapsed_str}: "
        f"Episode {episode} | "
        f"Mean(100): {last_100_episode_mean:.2f} | "
        f"epsilon: {eps:.2f}"
    )

    print("\n" + log_message)
    for line in lines:
        print(line)

    log_file.write(log_message + '\n')
    for line in lines:
        log_file.write(line + '\n')
    log_file.flush()

    # reset stats 
    for key in self.termination_stats:
        self.termination_stats[key] = 0
#==========================================================================================