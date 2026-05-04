from codebase.DDQN_PER_Dueling_agent.DDQN_PER_Dueling import DDQN_PER_Dueling
import sys
import os
import torch
import gymnasium as gym
from codebase.log_variable import RUNS_DIR, device
from codebase.AgentDDQN.DDQN import Agent
from codebase.AgentDDQN.DDQN import QNetwork

# def visualize_model(model_file=None, nb_episodes=1, seed=42):

    # print(model_file)

    # if model_file is None or not os.path.exists(model_file):
    #     exit(84)

    # checkpoint = torch.load(model_file, map_location=device)
    # state_dict = checkpoint.get("model_state_dict", checkpoint)

    # hyperparameters = checkpoint["hyperparameters"]

    # dql = Agent(hyperparameters=hyperparameters, nb_episode=nb_episodes, seed=seed)

    # env = gym.make(
    #     "LunarLander-v3",
    #     render_mode="human",
    #     continuous=dql.enable_continuous,
    #     gravity=dql.gravity,
    #     enable_wind=dql.enable_wind,
    #     wind_power=dql.wind_power,
    #     turbulence_power=dql.turbulence_power
    # )

    # num_states = env.observation_space.shape[0]
    # num_actions = env.action_space.n

    # policy_dqn = Agent(num_states, dql.number_of_layer, num_actions, dql.fc1_nodes).to(device)
    # policy_dqn.load_state_dict(state_dict)
    # policy_dqn.eval()

    # state, _ = env.reset(seed=seed)
    # for ep in range(nb_episodes):
    #     state, _ = env.reset()
    #     state = torch.tensor(state, dtype=torch.float, device=device)
    #     total_reward = 0.0
    #     terminated = False
    #     truncated = False
    #     steps = 0
    #     while not (terminated or truncated):
    #         with torch.no_grad():
    #             action = policy_dqn(state.unsqueeze(0)).squeeze().argmax().item()
    #         new_state, reward, terminated, truncated, _ = env.step(action)
    #         total_reward += reward
    #         state = torch.tensor(new_state, dtype=torch.float, device=device)
    #         steps += 1
    #         if steps > 5000:
    #             break
    #     print(f"Episode {ep + 1} finished, reward: {total_reward}")
    # env.close()
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