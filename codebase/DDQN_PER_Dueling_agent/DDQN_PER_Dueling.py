# IMPORT ===================================================
from torch import nn

from datetime import datetime, timedelta
from ..log_variable import device, RUNS_DIR, DATE_FORMAT
import gymnasium as gym
import yaml
import torch
import os
import random
import numpy as np
#========================================================

class DDQN_PER_Dueling():
    from .init_training import _setup_logging, _setup_training
    from .optimization import optimize
    from ..utils.plotting import save_graph
    from ..training import _training

    def __init__(self, hyperparameter_set, nb_episode=1):
        # Open hyperparameters
        with open('config/hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]
        self.hyperparameter_set = hyperparameter_set
        self.hyperparameters = hyperparameters

        # Other 
        self.nb_episode = nb_episode

        # === SHORTCUTS ===
        algo = hyperparameters["algo"]
        hypers = hyperparameters["hypers"]
        memory = hypers["memory"]
        epsilon_cfg = hypers["epsilon"]
        envs = hyperparameters["env"]
        winds = envs["wind"]
        gravity = envs["gravity"]
        fuel = envs["fuel"]
        continuous = envs["continuous"]


        # === Network ===
        self.seed = hyperparameters["seed"]
        self.fc1_nodes = hyperparameters["fc1_nodes"]
        self.number_of_layer = hyperparameters["number_of_layer"]

        # === Algo ===
        self.activation_function = algo["activation_function"]
        self.optimizer_function = algo["optimizer"] # change the name

        # === Env ===
        self.enable_wind = winds["enable_wind"]
        self.wind_power = winds["wind_power"]
        self.turbulence_power = winds["turbulence_power"]

        self.gravity = gravity["gravity"]

        self.enable_fuel = fuel["enable_fuel"]

        self.enable_continuous = continuous["enable_continuous"]

        # === HYPERS ===
        self.replay_memory_size = memory["replay_memory_size"]
        self.mini_batch_size    = memory["mini_batch_size"]

        self.epsilon_init  = epsilon_cfg["epsilon_init"]
        self.epsilon_decay = epsilon_cfg["epsilon_decay"]
        self.epsilon_min   = epsilon_cfg["epsilon_min"]

        self.learning_rate_a   = hypers["learning_rate_a"]
        self.network_sync_rate = hypers["network_sync_rate"]
        self.stop_on_reward    = hypers["stop_on_mean_reward"]
        
        # === Neural Network ===
        self.loss_fn = nn.MSELoss()
        self.optimizer = None

        # === Path to Run info ===
        self.LOG_FILE   = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}.png')

    def _training_update(self, episode, episode_reward, best_reward, rewards_per_episode,
                          epsilon_history, memory, policy_dqn, target_dqn, epsilon, step_count,
                          last_graph_update_time, log_message):
        # if not is_training:
        #     return best_reward, epsilon, step_count, last_graph_update_time

        current_time = datetime.now()
        if current_time - last_graph_update_time > timedelta(seconds=60):
            self.save_graph(rewards_per_episode, epsilon_history)
            last_graph_update_time = current_time

        # ===================== TRAINING STEP =====================
        if len(memory) > self.mini_batch_size and step_count % 4 == 0:
            mini_batch, indices, weights = memory.sample(self.mini_batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn, weights, indices, memory)

            epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
            epsilon_history.append(epsilon)

            step_count += 1

            if step_count > self.network_sync_rate:
                target_dqn.load_state_dict(policy_dqn.state_dict())
                step_count = 0

        return best_reward, epsilon, step_count, last_graph_update_time

    def _run_episode(self, env, episode, policy_dqn, memory,
                     epsilon, action_counts, step_count):

        # ===================== RESET VARIABLES =====================
        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)

        terminated = False
        episode_reward = 0.0

        max_steps = 1000
        steps = 0

        # ===================== LOOP START =====================
        while not terminated and steps < max_steps:
            steps += 1

            # ===================== ACTION SELECTION =====================
            if random.random() < epsilon or episode < 100:
                action = env.action_space.sample()
                action = torch.tensor(action, dtype=torch.int64, device=device)
            else:
                with torch.no_grad():
                    action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()
            # if is_training and (random.random() < epsilon or episode < 100):
            #     action = env.action_space.sample()
            #     action = torch.tensor(action, dtype=torch.int64, device=device)
            # else:
            #     with torch.no_grad():
            #         action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

            # ===================== STEP ENV =====================
            new_state, reward, terminated, _, _ = env.step(action.item())

            episode_reward += reward

            new_state = torch.tensor(new_state, dtype=torch.float, device=device)
            reward = torch.tensor(reward, dtype=torch.float, device=device)

            # ===================== MEMORY =====================
            memory.append((state, action, new_state, reward, terminated))
            step_count += 1

            # ===================== STATE UPDATE =====================
            state = new_state
            action_counts[action.item()] += 1

            # ===================== DEBUG =====================
            # print(f"episode : {episode} ----- action : {action_counts} ------ rewards : {episode_reward}")
            # print(f"y : {state[1]}")
            # ===================== END =====================
            if steps >= max_steps:
                terminated = True

        return episode_reward, step_count

    def run(self, render=False):
        #=====================================================================================
        # Log Message
        last_graph_update_time, log_message = self._setup_logging()
        # Create instance of the environment and agent
        env = gym.make("LunarLander-v3", render_mode="human" if render else "rgb_array", continuous=self.enable_continuous, gravity=self.gravity,
               enable_wind=self.enable_wind, wind_power=self.wind_power, turbulence_power=self.turbulence_power)
        # Setup training value or load best model
        policy_dqn, target_dqn, memory, epsilon, epsilon_history, step_count, best_reward = \
        self._setup_training(env)
        #=====================================================================================

        # Train variable
        rewards_per_episode = []
        action_counts = [0,0,0,0]

        last_100_episode_mean = -999
        episode = 0

        try:
            while (last_100_episode_mean < self.stop_on_reward and episode < self.nb_episode):

                episode_reward, step_count = self._run_episode(env, episode, policy_dqn,
                    memory, epsilon, action_counts, step_count)
                rewards_per_episode.append(episode_reward)

                # Update best model ===============================================================
                best_reward, epsilon, step_count, last_graph_update_time = self._training_update(
                    episode, episode_reward, best_reward, rewards_per_episode,
                    epsilon_history, memory, policy_dqn, target_dqn,
                    epsilon, step_count, last_graph_update_time, log_message
                )
                episode += 1
                if len(rewards_per_episode) >= 100:
                    last_100_episode_mean = np.mean(rewards_per_episode[-100:])

                if len(rewards_per_episode) >= 100 and episode % 100 == 0:
                    print(f"Episode {episode} | Mean(100): {last_100_episode_mean:.2f} | epsilon: {epsilon:.2f}")

                if last_100_episode_mean >= 200:
                    print(f"Solved in {episode} episodes!")
                    torch.save({
                        "model_state_dict": policy_dqn.state_dict(),
                        "hyperparameters": self.hyperparameters,
                    }, self.MODEL_FILE + "_solved")
                    break
                #===================================================================================
        except KeyboardInterrupt:
            torch.save({
                    "model_state_dict": policy_dqn.state_dict(),
                    "hyperparameters": self.hyperparameters,
                }, self.MODEL_FILE + "_last")
        #=======================================================================================















