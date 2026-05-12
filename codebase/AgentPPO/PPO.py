import numpy as np
import yaml
import os
import gymnasium as gym
import random
from collections import namedtuple, deque
from datetime import datetime, timedelta
import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical
from codebase.AgentPPO.model import ActorNetwork, CriticNetwork
from ..log_variable import device, RUNS_DIR, config_path, DATE_FORMAT, max_action, MAX_FUEL
from ..utils.function_finders import get_activation, get_optimizer
from ..utils.fuel_function import fuel_reward, fuel_end_reward

class PPOMemory:
    def __init__(self, batch_size):
        self.states = []
        self.probs = []
        self.vals = []
        self.actions = []
        self.rewards = []
        self.dones = []

        self.batch_size = batch_size

    def generate_batches(self):
        n_states = len(self.states)
        batch_start = np.arange(0, n_states, self.batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i+self.batch_size] for i in batch_start]

        return np.array(self.states),\
                np.array(self.actions),\
                np.array(self.probs),\
                np.array(self.vals),\
                np.array(self.rewards),\
                np.array(self.dones),\
                batches

    def store_memory(self, state, action, probs, vals, reward, done):
        self.states.append(state)
        self.actions.append(action)
        self.probs.append(probs)
        self.vals.append(vals)
        self.rewards.append(reward)
        self.dones.append(done)

    def clear_memory(self):
        self.states = []
        self.probs = []
        self.actions = []
        self.rewards = []
        self.dones = []
        self.vals = []

class AgentPPO():
    from codebase.utils.plotting import save_graph, plot_scores, classify_termination, log_with_termination_stats, update_termination_stats


    def __init__(self, n_epochs=10, hyperparameter_set=None, hyperparameters=None, nb_episode=20000, seed=42):
        self.gamma = 0.99
        self.policy_clip = 0.2
        self.n_epochs = n_epochs # replace by number of episodes or steps
        self.gae_lambda = 0.95
        self.alpha = 0.0003

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
        T.manual_seed(seed)
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
        self.loss_fn = T.nn.MSELoss()
        self.optimizer = None

        # === Path to Run info ===
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-PPO.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-PPO')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-PPO.png')

        self.termination_stats = {
            "landed": 0,
            "crash_high_speed": 0,
            "crash_bad_angle": 0,
            "out_of_bounds": 0,
            "time_limit": 0,
            "unknown": 0
        }

        self.n_steps = 0
        self.learn_iters = 0
        self.update_freq = 2048

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

        self.actor = ActorNetwork(self.action_size, self.state_size, self.alpha, self.activation_function, self.optimizer_function, self.nb_neuron, self.nb_layer)
        self.critic = CriticNetwork(self.state_size, self.alpha, self.activation_function, self.optimizer_function, self.nb_neuron, self.nb_layer)

        self.memory = PPOMemory(self.batch_size)

    def remember(self, state, action, probs, vals, reward, done):
        self.memory.store_memory(state, action, probs, vals, reward, done)

    def step(self, state, action, prob, val, reward, done):
        self.remember(state, action, prob, val, reward, done)
        self.n_steps += 1

        if self.n_steps % self.update_freq == 0:
            self.learn()
            self.learn_iters += 1

    def act(self, observation, fuel):
        if self.enable_fuel and fuel <= 0:
            return 0, 0.0, 0.0 # do nothing if no more fuel
        
        state = T.from_numpy(observation).float().unsqueeze(0).to(device)

        dist = self.actor(state)
        value = self.critic(state)
        action = dist.sample()

        probs = dist.log_prob(action).squeeze().item()
        action = action.squeeze().item()
        value = value.squeeze().item()

        return action, probs, value

    def learn(self):
        for _ in range(self.n_epochs):
            state_arr, action_arr, old_prob_arr, vals_arr,\
            reward_arr, dones_arr, batches = \
                    self.memory.generate_batches()

            values = vals_arr
            advantage = np.zeros(len(reward_arr), dtype=np.float32)

            advantage = np.zeros_like(reward_arr, dtype=np.float32)
            gae = 0
            values = np.append(vals_arr, 0)

            for t in reversed(range(len(reward_arr))):
                delta = reward_arr[t] + self.gamma * values[t+1] * (1 - dones_arr[t]) - values[t]
                gae = delta + self.gamma * self.gae_lambda * (1 - dones_arr[t]) * gae
                advantage[t] = gae
            advantage = (advantage - advantage.mean()) / (advantage.std() + 1e-8)

            advantage = T.tensor(advantage, dtype=T.float).to(device)
            values = T.tensor(vals_arr, dtype=T.float).to(device)
            for batch in batches:
                states = T.tensor(state_arr[batch], dtype=T.float).to(device)
                old_probs = T.tensor(old_prob_arr[batch], dtype=T.float).to(device)
                actions = T.tensor(action_arr[batch], dtype=T.long).to(device)

                dist = self.actor(states)
                critic_value = self.critic(states)

                critic_value = T.squeeze(critic_value)

                new_probs = dist.log_prob(actions)
                # prob_ratio = new_probs.exp() / old_probs.exp()
                prob_ratio = (new_probs - old_probs).exp()
                weighted_probs = advantage[batch] * prob_ratio
                weighted_clipped_probs = T.clamp(prob_ratio, 1-self.policy_clip,
                        1+self.policy_clip)*advantage[batch]
                actor_loss = -T.min(weighted_probs, weighted_clipped_probs).mean()

                returns = advantage[batch] + values[batch]
                critic_loss = (returns-critic_value)**2
                critic_loss = critic_loss.mean()

                total_loss = actor_loss + 0.5*critic_loss
                self.actor.optimizer.zero_grad()
                self.critic.optimizer.zero_grad()
                total_loss.backward()
                self.actor.optimizer.step()
                self.critic.optimizer.step()

        self.memory.clear_memory()

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

        # Train variable
        start_time = datetime.now()
        score = 0.0
        mean_rewards = []
        scores = []
        last_100_episode_mean = -999
        fuel = MAX_FUEL
        episode = 0

        try:
            while (episode < self.nb_episode):
                state, _ = env.reset()
                score = 0.0
                for _ in range(max_action):
                    action, prob, val = self.act(state, fuel)
                    next_state, reward, terminated, truncated, _ = env.step(action)
                    done = terminated or truncated
                    if self.enable_fuel:
                        reward, fuel = fuel_reward(action, reward, fuel)
                    self.step(state, action, prob, val, reward, done)

                    state = next_state
                    score += reward
                    if done:
                        if self.enable_fuel:
                            score += fuel_end_reward(fuel)
                        self.update_termination_stats(next_state, terminated, truncated)
                        break
                mean_rewards.append(score)
                scores.append(score)

                episode += 1
                if len(mean_rewards) >= 100:
                    last_100_episode_mean = np.mean(mean_rewards[-100:])

                if episode % 100 == 0:
                    self.log_with_termination_stats(
                        log_file,
                        episode,
                        last_100_episode_mean,
                        0,
                        last_graph_update_time,
                        start_time
                    )

                if last_100_episode_mean >= self.stop_on_reward:
                    self.log_with_termination_stats(
                        log_file,
                        episode,
                        last_100_episode_mean,
                        0,
                        last_graph_update_time,
                        start_time
                    )
                    log_message = f"Solved in {episode} episodes! With a score of {np.mean(mean_rewards[-100:])}"
                    print(log_message)
                    log_file.write(log_message + '\n')
                    log_file.flush()
                    T.save({
                        "model_state_dict": {
                            "actor": self.actor.state_dict(),
                            "critic": self.critic.state_dict()
                        },
                        "hyperparameters": self.hyperparameters,
                    }, self.MODEL_FILE + "_solved.pt")
                    break
        except KeyboardInterrupt:
            T.save({
                "model_state_dict": {
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict()
                },
                "hyperparameters": self.hyperparameters,
            }, self.MODEL_FILE + "_solved.pt")
        AgentPPO.plot_scores(scores, self.MODEL_FILE + "_detail.png")
        return mean_rewards