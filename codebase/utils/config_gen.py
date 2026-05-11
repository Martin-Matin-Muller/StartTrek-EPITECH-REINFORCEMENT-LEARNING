import itertools
import yaml
import copy

base_config = {
    "fc1_nodes": 32,
    "number_of_layer": 2,
    "algo": {
        "activation_function": "relu",
        "optimizer": "adam"
    },
    "env": {
        "wind": {
            "enable_wind": False,
            "wind_power": 0,
            "turbulence_power": 0
        },
        "gravity": {
            "gravity": -10.0
        },
        "fuel": {
            "enable_fuel": False
        },
        "continuous": {
            "enable_continuous": False
        }
    },
    "hypers": {
        "memory": {
            "replay_memory_size": 100000,
            "mini_batch_size": 128
        },
        "epsilon": {
            "epsilon_init": 0.9,
            "epsilon_min": 0.01,
            "epsilon_decay": 0.9995
        },
        "stop_on_mean_reward": 200,
        "learning_rate_a": 0.001
    }
}

search_space = {
    "hypers.memory.mini_batch_size": [64, 128, 256],
    "hypers.epsilon.epsilon_decay": [0.995, 0.999, 0.9995],
    "hypers.learning_rate_a": [1e-4, 3e-4, 1e-3],
    "fc1_nodes": [16, 32, 64],
    "number_of_layer": [2, 3],
    "activation_function": ["relu", "tanh", "leaky_relu"],
    "optimizer": ["adam", "sgd", "rmsprop"]
}

# === Helper to set nested dict values ===
def set_nested(config, key_path, value):
    keys = key_path.split(".")
    d = config
    for k in keys[:-1]:
        d = d[k]
    d[keys[-1]] = value

all_configs = {}

for i, values in enumerate(itertools.product(*search_space.values()), start=1):
    config = copy.deepcopy(base_config)

    for key, value in zip(search_space.keys(), values):
        set_nested(config, key, value)

    all_configs[f"LunarLander-multiple-v{i}"] = config

with open("config/hyperparameters_grid.yml", "w") as f:
    yaml.dump(all_configs, f, sort_keys=False)

print(f"{len(all_configs)} configurations generated.")