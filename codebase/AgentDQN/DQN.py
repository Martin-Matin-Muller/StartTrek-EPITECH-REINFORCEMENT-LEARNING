# IMPORT ===================================================
import numpy as np
import gymnasium as gym
import random
from collections import namedtuple, deque
from datetime import datetime, timedelta

from .model import QNetwork

import torch
import torch.nn.functional as F
import torch.optim as optim
import yaml
import os
from ..log_variable import device, RUNS_DIR, config_path, DATE_FORMAT, max_action, MAX_FUEL
from ..utils.function_finders import get_activation, get_optimizer
from ..utils.fuel_function import fuel_reward, fuel_end_reward
#========================================================

class AgentDQN():
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
        algo = hp["algo"]
        hypers = hp["hypers"]
        memory = hypers["memory"]
        epsilon_cfg = hypers["epsilon"]
        envs = hp["env"]
        winds = envs["wind"]
        gravity = envs["gravity"]
        fuel = envs["fuel"]
        continuous = envs["continuous"]

        # === Network ===
        self.nb_neuron = hp["fc1_nodes"]
        self.nb_layer = hp["number_of_layer"]

        # === Env ===
        self.enable_wind = winds["enable_wind"]
        self.wind_power = winds["wind_power"]
        self.turbulence_power = winds["turbulence_power"]

        self.gravity = gravity["gravity"]

        self.enable_fuel = fuel["enable_fuel"]

        self.enable_continuous = continuous["enable_continuous"]

        # === HYPERS ===
        self.replay_memory_size = memory["replay_memory_size"]
        self.batch_size = memory["mini_batch_size"]

        self.epsilon_init = epsilon_cfg["epsilon_init"]
        self.epsilon_decay = epsilon_cfg["epsilon_decay"]
        self.epsilon_min = epsilon_cfg["epsilon_min"]

        self.lr = hypers["learning_rate_a"]
        self.stop_on_reward = hypers["stop_on_mean_reward"]
        self.discount_factor_g = 0.99

        # === Algo ===
        self.activation_function = get_activation(algo["activation_function"])
        self.optimizer_function = get_optimizer(algo["optimizer"])


        # === Neural Network ===
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        # === Path to Run info ===
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-DQN.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-DQN')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-DQN.png')

        self.termination_stats = {
            "landed": 0,
            "crash_high_speed": 0,
            "crash_bad_angle": 0,
            "out_of_bounds": 0,
            "time_limit": 0,
            "unknown": 0
        }

    def step(self, state, action, reward, next_state, done):
        self.memory.add(state, action, reward, next_state, done)
        
        step_every = 4
        discount_factor = 0.99

        self.nb_step = (self.nb_step + 1) % step_every
        if self.nb_step == 0:
            if len(self.memory) > self.batch_size:
                experiences = self.memory.sample()
                self.learn(experiences, discount_factor)

    def act(self, state, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            action_values = self.qnetwork_local(state)
        self.qnetwork_local.train()

        # Epsilon-greedy action selection
        if random.random() > eps:
            return np.argmax(action_values.cpu().data.numpy())
        else:
            return random.choice(np.arange(self.action_size))

    def learn(self, experiences, gamma):
        states, actions, rewards, next_states, dones = experiences

        # Get max predicted Q values (for next states) from target model
        Q_targets_next = self.qnetwork_target(next_states).detach().max(1)[0].unsqueeze(1)
        # Compute Q targets for current states 
        Q_targets = rewards + (gamma * Q_targets_next * (1 - dones))

        # Get expected Q values from local model
        Q_expected = self.qnetwork_local(states).gather(1, actions)

        # Compute loss
        loss = F.mse_loss(Q_expected, Q_targets)
        # Minimize the loss
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # ------------------- update target network ------------------- #
        tau = 0.001
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau)                  

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_(tau*local_param.data + (1.0-tau)*target_param.data)



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

        self.qnetwork_local = QNetwork(self.state_size, self.action_size, self.seed, self.activation_function, self.nb_neuron, self.nb_layer).to(device)
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed, self.activation_function, self.nb_neuron, self.nb_layer).to(device)

        self.memory = ReplayBuffer(self.action_size, self.replay_memory_size, self.batch_size, self.seed)
        self.optimizer = self.optimizer_function(self.qnetwork_local.parameters(), lr=self.lr)

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
        # Create instance of the environment and agent
        env = gym.make("LunarLander-v3", render_mode="human" if visualization else "rgb_array",
                        continuous=self.enable_continuous, gravity=self.gravity,
                        enable_wind=self.enable_wind,
                        wind_power=self.wind_power, turbulence_power=self.turbulence_power)
        env.reset(seed=self.seed)
        # Setup training value or load best model
        self._setup_training(env)
        #=====================================================================================

        # Train variable
        start_time = datetime.now()
        score = 0.0
        mean_rewards = []
        scores = []
        last_100_episode_mean = -999
        episode = 0
        epsilon_history = []
        eps = self.epsilon_init   
        fuel = MAX_FUEL

        try:
            while (episode < self.nb_episode):
                state, _ = env.reset()
                score = 0
                for _ in range(max_action):
                    action = self.act(state, eps)
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
                        "model_state_dict": self.qnetwork_local.state_dict(),
                        "hyperparameters": self.hyperparameters,
                    }, self.MODEL_FILE + "_solved.pt")
                    break
        except KeyboardInterrupt:
            torch.save({
                    "model_state_dict": self.qnetwork_local.state_dict(),
                    "hyperparameters": self.hyperparameters,
                }, self.MODEL_FILE + "_last.pt")
        AgentDQN.plot_scores(scores, self.MODEL_FILE + "_detail.png")
        return mean_rewards


class ReplayBuffer:

    def __init__(self, action_size, buffer_size, batch_size, seed):
        self.action_size = action_size
        self.memory = deque(maxlen=buffer_size)  
        self.batch_size = batch_size
        self.experience = namedtuple("Experience", field_names=["state", "action", "reward", "next_state", "done"])
        self.seed = random.seed(seed)
    
    def add(self, state, action, reward, next_state, done):
        e = self.experience(state, action, reward, next_state, done)
        self.memory.append(e)
    
    def sample(self):
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.from_numpy(np.vstack([e.state for e in experiences if e is not None])).float().to(device)
        actions = torch.from_numpy(np.vstack([e.action for e in experiences if e is not None])).long().to(device)
        rewards = torch.from_numpy(np.vstack([e.reward for e in experiences if e is not None])).float().to(device)
        next_states = torch.from_numpy(np.vstack([e.next_state for e in experiences if e is not None])).float().to(device)
        dones = torch.from_numpy(np.vstack([e.done for e in experiences if e is not None]).astype(np.uint8)).float().to(device)
  
        return (states, actions, rewards, next_states, dones)

    def __len__(self):
        return len(self.memory)