import itertools
import yaml
import copy

base_config = {
    "replay_memory_size": 100000,
    "epsilon_init": 0.9,
    "epsilon_min": 0.01,
    "stop_on_reward": 300,
    "enable_double_dqn": True,
    "enable_dueling_dqn": True,
    "discount_factor_g": 0.99,
}

search_space = {
    "mini_batch_size": [64, 128, 256, 512, 1024],
    "epsilon_decay": [0.995, 0.999, 0.9995, 0.9997],
    "learning_rate_a": [1e-4, 3e-4, 1e-3, 5e-4],
    "network_sync_rate": [250, 500, 1000],
    "fc1_nodes": [8, 16, 32],
}

all_configs = {}

for i, values in enumerate(itertools.product(*search_space.values()), start=1):
    config = copy.deepcopy(base_config)

    for key, value in zip(search_space.keys(), values):
        config[key] = value

    all_configs[f"LunarLander-multiple-v{i}"] = config

with open("config/hperparameters_grid.yml", "w") as f:
    yaml.dump(all_configs, f, sort_keys=False)

print(f"{len(all_configs)} configurations générées.")