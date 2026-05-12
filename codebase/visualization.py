import os
import torch
import gymnasium as gym
import sys

from codebase.log_variable import device

from .utils.function_finders import get_activation

from codebase.AgentDQN.model import QNetwork as DQN_Network
from codebase.AgentDDQN.model import QNetwork as DDQN_Network
from codebase.AgentDDQN_PER.model import QNetwork as DDQN_PER_Network
from codebase.AgentDDQN_PER_Dueling.model import QNetwork as DDQN_PER_Dueling_Network
from codebase.AgentDDQN_PER_Dueling_N_Step.model import QNetwork as DDQN_PER_Dueling_N_Step_Network
from codebase.AgentDDQN_PER_Dueling_N_Step_Noisy.model import QNetwork as DDQN_PER_Dueling_N_Step_Noisy_Network
from codebase.AgentRainbowDQN.model import QNetwork as Rainbow_Network
from codebase.AgentPPO.model import ActorNetwork as PPO_Network


MODEL_MAP = {
    "Rainbow": Rainbow_Network,
    "DDQN_PER_Dueling_N_Step_Noisy": DDQN_PER_Dueling_N_Step_Noisy_Network,
    "DDQN_PER_Dueling_N_Step": DDQN_PER_Dueling_N_Step_Network,
    "DDQN_PER_Dueling": DDQN_PER_Dueling_Network,
    "DDQN_PER": DDQN_PER_Network,
    "DDQN": DDQN_Network,
    "DQN": DQN_Network,
    "PPO": PPO_Network,
}


def visualize_model(model_path: str, nb_episodes: int, seed: int):

    if not os.path.isfile(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")

    checkpoint = torch.load(model_path, map_location=device)

    if "model_state_dict" not in checkpoint or "hyperparameters" not in checkpoint:
        raise ValueError("Invalid checkpoint format.")

    config = checkpoint["hyperparameters"]

    env_config = config["env"]
    wind = env_config["wind"]
    gravity = env_config["gravity"]

    env = gym.make(
        "LunarLander-v3",
        render_mode="human",
        enable_wind=wind["enable_wind"],
        wind_power=wind["wind_power"],
        turbulence_power=wind["turbulence_power"],
        gravity=gravity["gravity"],
    )

    env.action_space.seed(seed)

    state_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    model_name = None
    for key in MODEL_MAP:
        if key in os.path.basename(model_path):
            model_name = key
            break

    if model_name is None:
        raise ValueError(f"Could not determine model type from path: {model_path}")

    ModelClass = MODEL_MAP[model_name]

    if model_name == "PPO":
        model = ModelClass(
            n_actions=action_dim,
            input_dims=state_dim,
            alpha=0.0,
            activation=get_activation(config["algo"]["activation_function"]),
            optimizer=torch.optim.Adam,
            hidden_dim=config["fc1_nodes"],
            nb_layer=config["number_of_layer"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"]["actor"])

    else:
        model = ModelClass(
            state_size=state_dim,
            action_size=action_dim,
            seed=seed,
            activation=get_activation(config["algo"]["activation_function"]),
            hidden_layer_neuron=config["fc1_nodes"],
            nb_layer=config["number_of_layer"],
        ).to(device)

        model.load_state_dict(checkpoint["model_state_dict"])

    model.eval()

    max_steps = 1000

    for episode in range(nb_episodes):

        state, _ = env.reset(seed=seed + episode)
        total_reward = 0
        done = False

        for step in range(max_steps):

            state_tensor = torch.from_numpy(state).float().unsqueeze(0).to(device)

            with torch.no_grad():
                if model_name == "PPO":
                    dist = model(state_tensor)
                    action = torch.argmax(dist.probs, dim=1).item()

                elif model_name == "Rainbow":
                    dist = model(state_tensor)
                    support = torch.linspace(-300, 300, 51).to(device)
                    q_values = (dist * support).sum(dim=2)
                    action = q_values.argmax(dim=1).item()

                else:
                    action = model(state_tensor).argmax(dim=1).item()

            next_state, reward, terminated, truncated, _ = env.step(action)

            total_reward += reward
            state = next_state

            if terminated or truncated:
                break

        print(f"Episode {episode + 1}: reward = {total_reward:.2f}")

    env.close()

if __name__ == "__main__":

    if len(sys.argv) != 4:
        print("Usage : python3 -m codebase.visualization <model_path> <nb_episode> <seed>")
        exit(84)

    model_path = sys.argv[1]

    try:
        nb_episodes = int(sys.argv[2])
        seed = int(sys.argv[3])
    except ValueError:
        print("Error: <nb_episode> and <seed> must be numbers.")
        exit(84)

    if nb_episodes <= 0:
        print("Error: <nb_episode> must be > 0.")
        exit(84)

    visualize_model(
        model_path=model_path,
        nb_episodes=nb_episodes,
        seed=seed
    )