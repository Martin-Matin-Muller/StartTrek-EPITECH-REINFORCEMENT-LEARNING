import os
import sys
import torch
import gymnasium as gym
from gymnasium.wrappers import RecordVideo
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pandas as pd

from codebase.DDQN_PER_Dueling_agent.dqn import DQN as DDQN_PER_DUELING_network
# import new model when implemented

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def evaluate_all_models(runs_dir="runs", nb_episode=100, seed=42):
    results = {}

    np.random.seed(seed)
    torch.manual_seed(seed)

    for file in os.listdir(runs_dir):
        if not file.endswith(".pt"):
            continue

        print(f"\nEvaluating {file}")

        checkpoint = torch.load(os.path.join(runs_dir, file), map_location=device)

        if "model_state_dict" not in checkpoint or "hyperparameters" not in checkpoint:
            print(f"Skipping {file} (invalid checkpoint)")
            continue

        config = checkpoint["hyperparameters"]

        env = gym.make("LunarLander-v3", render_mode="rgb_array") # TODO: add les autre paramètre 

        model_name = file.replace(".pt", "")
        video_path = os.path.join(runs_dir, "videos", model_name)

        env = RecordVideo(
            env,
            video_folder=video_path,
            episode_trigger=lambda ep: ep > nb_episode - 4,  # record last 3 episodes
            name_prefix=model_name
        )

        env.action_space.seed(seed)

        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        # --- Model selection ---
        if "DDQN-PER-Dueling" in file:
            model = DDQN_PER_DUELING_network(
                state_dim=state_dim,
                number_layer=config["number_of_layer"],
                action_dim=action_dim,
                hidden_dim=config["fc1_nodes"]
            ).to(device)
        # Add other option when implemented
        else:
            print(f"Skipping unknown model type: {file}")
            env.close()
            continue

        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        rewards = []

        # --- Evaluation loop ---
        for episode in range(nb_episode):
            state, _ = env.reset(seed=seed + episode)
            state = torch.from_numpy(state).float().to(device)

            episode_reward = 0.0

            with torch.no_grad():
                for step in range(1000):
                    action = model(state.unsqueeze(0)).argmax(dim=1).item()

                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated

                    episode_reward += reward
                    state = torch.from_numpy(next_state).float().to(device)

                    if done:
                        break

            rewards.append(episode_reward)

        # --- Metrics ---
        rewards = np.array(rewards)

        results[file] = {
            "mean": float(np.mean(rewards)),
            "std": float(np.std(rewards)),
            "median": float(np.median(rewards)),
            "min": float(np.min(rewards)),
            "max": float(np.max(rewards)),
            "success_rate": float(np.mean(rewards >= 200))
        }

        env.close()

    # ================= RESULTS =================
    print("\n=== RESULTS ===")
    for model_name, metrics in results.items():
        print(model_name, metrics)

    # --- Plot ---
    sorted_items = sorted(results.items(), key=lambda x: x[1]["mean"], reverse=True)

    model_files = [k for k, _ in sorted_items]
    model_names = [k.replace(".pt", "") for k in model_files]

    means = [v["mean"] for _, v in sorted_items]
    stds = [v["std"] for _, v in sorted_items]
    success_rates = [v["success_rate"] for _, v in sorted_items]

    plt.figure(figsize=(15, 5))

    x = np.arange(len(model_names))

    plt.subplot(1, 3, 1)
    plt.bar(x, means)
    plt.title("Mean Reward")
    plt.xticks(x, model_names, rotation=30, ha='right')

    plt.subplot(1, 3, 2)
    plt.bar(x, stds)
    plt.title("Std Dev")
    plt.xticks(x, model_names, rotation=30, ha='right')

    plt.subplot(1, 3, 3)
    plt.bar(x, success_rates)
    plt.title("Success Rate")
    plt.xticks(x, model_names, rotation=30, ha='right')

    plt.tight_layout()
    plt.savefig(os.path.join(runs_dir, "comparison.png"))
    plt.close()

    # --- CSV ---
    df = pd.DataFrame(results).T.reset_index()
    df.columns = ["model", "mean", "std", "median", "min", "max", "success_rate"]

    df = df.sort_values("mean", ascending=False)
    df.to_csv(os.path.join(runs_dir, "results.csv"), index=False)

    return results


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("Not enough argument : python3 -m codebase.eval <nb_episode> <seed>")
        exit(84)

    try:
        nb_episode = int(sys.argv[1])
        seed = int(sys.argv[2])
    except ValueError:
        print("Usage : python3 -m codebase.eval <nb_episode> <seed>")
        print("<nb_episode> and <seed> must be valid number")
        exit(84)

    evaluate_all_models(
        nb_episode=nb_episode if nb_episode > 4 else 10,
        seed=seed if seed > 0 else 42
    )