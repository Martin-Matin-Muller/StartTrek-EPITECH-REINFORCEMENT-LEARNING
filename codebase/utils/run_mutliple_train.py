import subprocess
import yaml

with open("config/hyperparameters_grid.yml", "r") as f:
    configs = yaml.safe_load(f)

for config_name in configs.keys():
    print(f"\n===== Training {config_name} =====")

    subprocess.run([
        "python",
        "codebase/agent.py", # NEED TO CHANGE
        config_name,
        "--train",
        "10000"
    ], check=True)

    print(f"===== Finished {config_name} =====\n")