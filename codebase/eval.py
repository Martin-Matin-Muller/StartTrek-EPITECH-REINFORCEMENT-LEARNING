import os
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd
from basic_agent.dqn import DQN

device = 'cuda' if torch.cuda.is_available() else 'cpu'

# NEED REWORK

if __name__ == '__main__':

    nb_episode = 100
    results = {}

    env = gym.make("LunarLander-v3", render_mode="rgb_array")


    env = RecordVideo(
        env,
        video_folder="runs/videos",
        episode_trigger=lambda e: True  # record every episode # pas faire ça
    )
    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    for file in os.listdir("runs/"):
        if file.endswith(".pt"):

            print(f"Evaluating {file}")

            checkpoint = torch.load("runs/" + file, map_location=device)

            config = checkpoint["hyperparameters"]

            model = DQN(
                state_dim=state_dim,
                number_layer=config["number_of_layer"],
                action_dim=action_dim,
                hidden_dim=config["fc1_nodes"],
                enable_dueling_dqn=config["enable_dueling_dqn"]
            ).to(device)

            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            rewards = []

            for episode in range(nb_episode):
                state, _ = env.reset()
                state = torch.tensor(state, dtype=torch.float32, device=device)

                episode_reward = 0.0

                for step in range(1000):
                    with torch.no_grad():
                        action = model(state.unsqueeze(0)).squeeze().argmax().item()

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    state = torch.tensor(next_state, dtype=torch.float32, device=device)

                    if done:
                        break

                rewards.append(episode_reward)

            mean_reward = np.mean(rewards)
            std_reward = np.std(rewards)
            success_rate = np.mean([r >= 200 for r in rewards])

            results[file] = {
                "mean": mean_reward,
                "std": std_reward,
                "success_rate": success_rate
            }

    print("\n=== RESULTS ===")
    for model_name, metrics in results.items():
        print(model_name, metrics)


    model_names = list(results.keys())

    means = [results[m]["mean"] for m in model_names]
    stds = [results[m]["std"] for m in model_names]
    success_rates = [results[m]["success_rate"] for m in model_names]
    plt.figure(figsize=(15, 5))

    # Mean reward
    plt.subplot(1, 3, 1)
    plt.bar(model_names, means)
    plt.title("Mean Reward")
    plt.xticks(rotation=45)

    # Std deviation
    plt.subplot(1, 3, 2)
    plt.bar(model_names, stds)
    plt.title("Std Dev")
    plt.xticks(rotation=45)

    # Success rate
    plt.subplot(1, 3, 3)
    plt.bar(model_names, success_rates)
    plt.title("Success Rate")
    plt.xticks(rotation=45)

    plt.tight_layout()
    plt.savefig("runs/comparison.png")
    plt.close()

    df = pd.DataFrame(results).T

    df = df.reset_index()
    df.columns = ["model", "mean", "std", "success_rate"]

    df['mean'] = df['mean'].apply(lambda x: '{:,.2f}'.format(x))
    df['std'] = df['std'].apply(lambda x: '{:,.2f}'.format(x))
    df['success_rate'] = df['success_rate'].apply(lambda x: '{:,.2f}'.format(x))

    df = df.sort_values("mean", ascending=False)
    df.to_csv("runs/results.csv")