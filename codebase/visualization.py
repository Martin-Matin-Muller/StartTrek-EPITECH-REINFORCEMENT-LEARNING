import sys
import os
import torch
import gymnasium as gym
from codebase.log_variable import RUNS_DIR, device
from codebase.AgentDDQN.DDQN import Agent
from codebase.AgentDDQN.DDQN import QNetwork

def visualize_model(model_path, nb_episodes=5, seed=42):
    # Charger le checkpoint
    checkpoint = torch.load(model_path, map_location=device)

    # Récupérer les hyperparamètres sauvegardés
    hyperparameters = checkpoint["hyperparameters"]
    state_dict = checkpoint["model_state_dict"]

    # Recréer l'agent avec les mêmes hyperparamètres
    agent = Agent(
        hyperparameters=hyperparameters,
        nb_episode=nb_episodes,
        seed=seed
    )

    # Créer l'environnement identique à l'entraînement
    env = gym.make(
        "LunarLander-v3",
        render_mode="human",
        continuous=agent.enable_continuous,
        gravity=agent.gravity,
        enable_wind=agent.enable_wind,
        wind_power=agent.wind_power,
        turbulence_power=agent.turbulence_power
    )

    state_size = env.observation_space.shape[0]
    action_size = env.action_space.n

    # Recréer le réseau
    model = QNetwork(
        state_size,
        action_size,
        seed,
        agent.nb_neuron,
        agent.nb_layer
    ).to(device)

    # Charger les poids
    model.load_state_dict(state_dict)
    model.eval()
    for episode in range(nb_episodes):
        state, _ = env.reset(seed=seed + episode)
        done = False
        total_reward = 0

        while not done:
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)

            with torch.no_grad():
                action_values = model(state_tensor)

            action = torch.argmax(action_values).item()

            state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            total_reward += reward

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()

if __name__ == '__main__':
    if len(sys.argv) != 4:
        print("Not enough argument : python3 -m codebase.visualization <model_name> <nb_episode> <seed>")
        exit(84)

    try:
        nb_episode = int(sys.argv[2])
        seed = int(sys.argv[3])
    except ValueError:
        print("Usage : python3 -m codebase.visualization <model_name> <nb_episode> <seed>")
        print("<nb_episode> and <seed> must be valid number")
        exit(84)

    visualize_model(
        model_path=sys.argv[1],
        nb_episodes=nb_episode if nb_episode > 0 else 10,
        seed=seed if seed > 0 else 42
    )