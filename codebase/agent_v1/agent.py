# IMPORT ===================================================
from torch import nn
from .log_variable import RUNS_DIR

import gymnasium as gym
import yaml
import torch
import os
#========================================================

class Agent():
    from .evaluation import _evaluation
    from .init_training import _setup_logging, _setup_training
    from .optimization import optimize
    from ..utils.plotting import save_graph
    from .training import _training

    def __init__(self, hyperparameter_set, nb_episode=1):
        # Open hyperparameters
        with open('config/hyperparameters.yml', 'r') as file:
        # with open('config/hyperparameters_grid.yml', 'r') as file: #=====================================================TEST
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        self.hyperparameters = hyperparameters

        # Other 
        self.nb_episode = nb_episode

        # Hyperparameters
        self.env_id             = hyperparameters['env_id']
        self.learning_rate_a    = hyperparameters['learning_rate_a']
        self.discount_factor_g  = hyperparameters['discount_factor_g']
        self.network_sync_rate  = hyperparameters['network_sync_rate']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size    = hyperparameters['mini_batch_size']
        self.epsilon_init       = hyperparameters['epsilon_init']
        self.epsilon_decay      = hyperparameters['epsilon_decay']
        self.epsilon_min        = hyperparameters['epsilon_min']
        self.stop_on_reward     = hyperparameters['stop_on_reward'] # to implement
        self.fc1_nodes          = hyperparameters['fc1_nodes']
        self.enable_double_dqn  = hyperparameters['enable_double_dqn']
        self.enable_dueling_dqn = hyperparameters['enable_dueling_dqn']
        # Neural Network
        self.loss_fn = nn.MSELoss()
        self.optimizer = None
        # Path to Run info
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')


    def run(self, is_training=True, render=False):
        #=====================================================================================
        # Log Message
        last_graph_update_time, log_message = self._setup_logging(is_training)
        # Create instance of the environment and agent
        env = gym.make("LunarLander-v3", render_mode="human" if render else "rgb_array")
        # Setup training value or load best model
        policy_dqn, target_dqn, memory, epsilon, epsilon_history, step_count, best_reward = \
        self._setup_training(is_training, env)
        #=====================================================================================

        # Train variable
        rewards_per_episode = []
        action_counts = [0,0,0,0]

        try:
            for episode in range(self.nb_episode):

                # Train
                episode_reward, step_count = self._training(env, is_training, episode, policy_dqn,
                    memory, epsilon, action_counts, step_count)
                rewards_per_episode.append(episode_reward)

                # Update best model
                best_reward, epsilon, step_count, last_graph_update_time = self._evaluation(
                    is_training, episode, episode_reward, best_reward, rewards_per_episode,
                    epsilon_history, memory, policy_dqn, target_dqn,
                    epsilon, step_count, last_graph_update_time, log_message
                )
        except KeyboardInterrupt:
            torch.save({
                    "model_state_dict": policy_dqn.state_dict(),
                    "hyperparameters": self.hyperparameters,
                }, self.MODEL_FILE + "_last")
        #=======================================================================================















