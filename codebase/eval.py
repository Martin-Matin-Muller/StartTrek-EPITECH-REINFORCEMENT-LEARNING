import os
import torch
import gymnasium as gym
import numpy as np
from dqn import DQN  # IMPORTANT

device = 'cuda' if torch.cuda.is_available() else 'cpu'

if __name__ == '__main__':

    nb_episode = 100
    results = {}

    env = gym.make("LunarLander-v3", render_mode="rgb_array")

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    for file in os.listdir("runs/"):
        if file.endswith(".pt"):

            print(f"Evaluating {file}")

            checkpoint = torch.load("runs/" + file, map_location=device)

            config = checkpoint["hyperparameters"]

            model = DQN(
                state_dim,
                action_dim,
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