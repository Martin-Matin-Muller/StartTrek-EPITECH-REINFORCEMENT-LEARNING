# IMPORT ===================================================
import numpy as np
import gymnasium as gym
import random
import torch
from datetime import datetime, timedelta

import yaml
import os
from ..log_variable import RUNS_DIR, config_path, DATE_FORMAT, max_action, MAX_FUEL
from ..utils.fuel_function import fuel_reward, fuel_end_reward
# ==========================================================

class AgentQLearning():
    from ..utils.plotting import save_graph, plot_scores, classify_termination, log_with_termination_stats, update_termination_stats

    def __init__(self, hyperparameter_set=None, hyperparameters=None, nb_episode=20000, seed=42):

        if hyperparameters is not None:
            self.hyperparameters = hyperparameters
            self.hyperparameter_set = "loaded_model"
        else:
            if not os.path.isfile(config_path):
                print(f"File '{config_path}' not found")
                exit(84)

            with open(config_path, "r") as file:
                all_hyperparameter_sets = yaml.safe_load(file)

            if hyperparameter_set not in all_hyperparameter_sets:
                print(f"Model '{hyperparameter_set}' not found")
                exit(84)

            self.hyperparameter_set = hyperparameter_set
            self.hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.nb_episode = nb_episode
        self.seed = seed
        random.seed(seed)
        np.random.seed(seed)

        hp = self.hyperparameters

        # === SHORTCUTS ===
        hypers = hp["hypers"]
        epsilon_cfg = hypers["epsilon"]
        envs = hp["env"]
        winds = envs["wind"]
        gravity = envs["gravity"]
        fuel = envs["fuel"]
        continuous = envs["continuous"]

        # === Env ===
        self.enable_wind = winds["enable_wind"]
        self.wind_power = winds["wind_power"]
        self.turbulence_power = winds["turbulence_power"]
        self.gravity = gravity["gravity"]
        self.enable_fuel = fuel["enable_fuel"]
        self.enable_continuous = continuous["enable_continuous"]

        # === HYPERS ===
        self.epsilon_init = epsilon_cfg["epsilon_init"]
        self.epsilon_decay = epsilon_cfg["epsilon_decay"]
        self.epsilon_min = epsilon_cfg["epsilon_min"]
        self.lr = hypers["learning_rate_a"]
        self.stop_on_reward = hypers["stop_on_mean_reward"]
        self.discount_factor_g = 0.99

        # === Q-table ===
        self.n_bins = 6

        self.state_bounds = [
            (-1.5,  1.5),
            (-1.5,  1.5),
            (-3.0,  3.0),
            (-3.0,  3.0),
            (-3.14, 3.14),
            (-5.0,  5.0),
            ( 0.0,  1.0),
            ( 0.0,  1.0),
        ]

        # === Path to Run info ===
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-QLearning.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-QLearning')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-QLearning.png')

        self.termination_stats = {
            "landed": 0,
            "crash_high_speed": 0,
            "crash_bad_angle": 0,
            "out_of_bounds": 0,
            "time_limit": 0,
            "unknown": 0
        }

    def _discretize(self, state):
        indices = []
        for val, (low, high) in zip(state, self.state_bounds):
            val = np.clip(val, low, high)
            bin_idx = int((val - low) / (high - low) * self.n_bins)
            bin_idx = min(bin_idx, self.n_bins - 1)
            indices.append(bin_idx)
        return tuple(indices)

    def step(self, state, action, reward, next_state, done):
        self.learn(state, action, reward, next_state, done)

    def act(self, state, fuel, eps=0.):
        if self.enable_fuel and fuel <= 0:
            return 0

        if random.random() < eps:
            return random.randint(0, self.n_actions - 1)

        disc = self._discretize(state)
        return int(np.argmax(self.q_table[disc]))

    def learn(self, state, action, reward, next_state, done):
        s = self._discretize(state)
        s_ = self._discretize(next_state)

        current_q = self.q_table[s][action]
        target_q = reward + (1 - float(done)) * self.discount_factor_g * np.max(self.q_table[s_])

        self.q_table[s][action] += self.lr * (target_q - current_q)

    def _setup_logging(self, log_file):
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        log_file.write(log_message + '\n')
        log_file.flush()
        return last_graph_update_time, log_message

    def _setup_training(self, env):
        self.n_actions = env.action_space.n
        self.state_size = env.observation_space.shape[0]

        table_shape = tuple([self.n_bins] * self.state_size) + (self.n_actions,)
        self.q_table = np.zeros(table_shape)

    def _training_update(self, rewards_per_episode,
                         epsilon_history, epsilon, last_graph_update_time):

        epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
        epsilon_history.append(epsilon)

        current_time = datetime.now()
        if current_time - last_graph_update_time > timedelta(seconds=60):
            self.save_graph(rewards_per_episode, epsilon_history)
            last_graph_update_time = current_time

        return last_graph_update_time, epsilon

    def run(self, visualization=False):
        #=====================================================================================
        # Log Message
        log_file = open(self.LOG_FILE, 'w')
        last_graph_update_time, log_message = self._setup_logging(log_file)
        # Create instance of the environment
        env = gym.make("LunarLander-v3", render_mode="human" if visualization else "rgb_array",
                        continuous=self.enable_continuous, gravity=self.gravity,
                        enable_wind=self.enable_wind,
                        wind_power=self.wind_power, turbulence_power=self.turbulence_power)
        env.reset(seed=self.seed)
        # Setup training
        self._setup_training(env)
        #======================================================A===============================

        start_time = datetime.now()
        scores = []
        mean_rewards = []
        last_100_episode_mean = -999
        episode = 0
        epsilon_history = []
        eps = self.epsilon_init
        fuel = MAX_FUEL

        try:
            while episode < self.nb_episode:
                state, _ = env.reset()
                score = 0
                fuel = MAX_FUEL

                for _ in range(max_action):
                    action = self.act(state, fuel, eps)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    if self.enable_fuel:
                        reward, fuel = fuel_reward(action, reward, fuel)
                    self.step(state, action, reward, next_state, done)
                    state = next_state
                    score += reward
                    if done:
                        if self.enable_fuel:
                            score += fuel_end_reward(fuel)
                        self.update_termination_stats(next_state, terminated, truncated)
                        break

                mean_rewards.append(score)
                scores.append(score)
                last_graph_update_time, eps = self._training_update(
                    mean_rewards, epsilon_history, eps, last_graph_update_time
                )

                episode += 1
                if len(mean_rewards) >= 100:
                    last_100_episode_mean = np.mean(mean_rewards[-100:])
                if episode % 100 == 0:
                    self.log_with_termination_stats(
                        log_file,
                        episode,
                        last_100_episode_mean,
                        eps,
                        last_graph_update_time,
                        start_time
                    )
                if last_100_episode_mean >= self.stop_on_reward:
                    self.log_with_termination_stats(
                        log_file,
                        episode,
                        last_100_episode_mean,
                        eps,
                        last_graph_update_time,
                        start_time
                    )
                    log_message = f"Solved in {episode} episodes! With a score of {np.mean(mean_rewards[-100:])}"
                    print(log_message)
                    log_file.write(log_message + '\n')
                    log_file.flush()
                    break

        except KeyboardInterrupt:
            torch.save({
                    "model_state_dict": self.qnetwork_local.state_dict(),
                    "hyperparameters": self.hyperparameters,
                }, self.MODEL_FILE + "_last.pt")

        AgentQLearning.plot_scores(scores, self.MODEL_FILE + "detail.png")
        return scores