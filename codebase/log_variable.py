import matplotlib
import torch
import os

# LOG ===================================================
DATE_FORMAT = "%m-%d %H:%M:%S"              # For printing date and time
# RUNS_DIR = "multiple_runs"  #=====================================================TEST
RUNS_DIR = "runs"                           # Directory for saving run info
os.makedirs(RUNS_DIR, exist_ok=True)
matplotlib.use('Agg') # generate plots as images and save them to a file instead of rendering to screen
device = 'cuda' if torch.cuda.is_available() else 'cpu'
config_path = "config/hyperparameters.yml"
max_action = 1000
#========================================================