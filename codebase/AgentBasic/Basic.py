# IMPORT ===================================================
import numpy as np
import gymnasium as gym
import random
from datetime import datetime, timedelta

import torch
import yaml
import os
from ..log_variable import RUNS_DIR, config_path, DATE_FORMAT, max_action
#========================================================

class AgentBasic():
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
        torch.manual_seed(seed)
        self.nb_step = 0

        hp = self.hyperparameters

        # === SHORTCUTS ===
        hypers = hp["hypers"]
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
        self.stop_on_reward = hypers["stop_on_mean_reward"]

        # === Path to Run info ===
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-Basic.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-Basic')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-Basic.png')

        self.termination_stats = {
            "landed": 0,
            "crash_high_speed": 0,
            "crash_bad_angle": 0,
            "out_of_bounds": 0,
            "time_limit": 0,
            "unknown": 0
        }

    def act(self, state):
        x = state[0]
        y = state[1]
        vx = state[2]
        vy = state[3]
        angle = state[4]
        angular_vel = state[5]

        # trop pencher sur la gauche
        if angle > 0.1:
            return 3  # moteur droite
        # trop pencher sur la droite
        if angle < -0.1:
            return 1  # moteur gauche

        # tombe trop
        if vy < -0.5:
            return 2  # moteur principal

        # trop à gauche
        if x < -0.1: # moteur droite
            return 3

        # trop à droite 
        if x > 0.1:
            return 1 # moteur gauche
        return 0


    def _setup_logging(self, log_file):
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        log_file.write(log_message + '\n')
        log_file.flush()
        return last_graph_update_time, log_message

    def _setup_training(self, env):
        self.action_size = env.action_space.n
        self.state_size = env.observation_space.shape[0]

    def _training_update(self, rewards_per_episode,
                      epsilon_history, epsilon, last_graph_update_time):

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
        # Create instance of the environment and agent
        env = gym.make("LunarLander-v3", render_mode="human" if visualization else "rgb_array",
                        continuous=self.enable_continuous, gravity=self.gravity,
                        enable_wind=self.enable_wind,
                        wind_power=self.wind_power, turbulence_power=self.turbulence_power)
        env.reset(seed=self.seed)
        # Setup training value or load best model
        self._setup_training(env)
        #=====================================================================================

        start_time = datetime.now()
        score = 0.0
        scores = []
        mean_rewards = []
        last_100_episode_mean = -999
        episode = 0
        epsilon_history = []
        eps = 0

        try:
            while (episode < self.nb_episode):
                state, _ = env.reset()
                score = 0
                for _ in range(max_action):
                    action = self.act(state)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    state = next_state
                    score += reward
                    if done:
                        self.update_termination_stats(next_state, terminated, truncated)
                        break
                mean_rewards.append(score)
                scores.append(score)
                last_graph_update_time, eps = self._training_update(mean_rewards, epsilon_history, eps, last_graph_update_time)


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
                    torch.save({
                        "hyperparameters": self.hyperparameters,
                    }, self.MODEL_FILE + "_solved.pt")
                    break
        except KeyboardInterrupt:
            torch.save({
                    "hyperparameters": self.hyperparameters,
                }, self.MODEL_FILE + "_last.pt")
        AgentBasic.plot_scores(scores, self.MODEL_FILE + "_detail.png")
        return scores
