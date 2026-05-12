# IMPORT ===================================================
import numpy as np
import gymnasium as gym
import random
from collections import namedtuple, deque
from datetime import datetime, timedelta

from .model import QNetwork
from collections import namedtuple
import torch
import torch.nn.functional as F
import torch.optim as optim
from torch.nn.utils import clip_grad_norm_
import yaml
import os
from ..log_variable import device, RUNS_DIR, config_path, DATE_FORMAT, max_action, MAX_FUEL
from ..utils.function_finders import get_activation, get_optimizer
from ..utils.fuel_function import fuel_reward, fuel_end_reward
#========================================================

class AgentRainbowDQN():
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

        self.per_alpha       = 0.7
        self.per_beta_start  = 0.5
        self.per_beta_frames = 150_000
        self.n_step = 3

        # ADD C51 (Distributional RL)
        self.n_atoms = 51
        self.v_min   = -300   # score minimum possible sur LunarLander
        self.v_max   =  300   # score maximum possible

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

        # === Algo ===
        self.activation_function = get_activation(algo["activation_function"])
        self.optimizer_function = get_optimizer(algo["optimizer"])

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

        # === Neural Network ===
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = None

        # === Path to Run info ===
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-RainbowDQN.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-RainbowDQN')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{self.hyperparameter_set}-RainbowDQN.png')

        self.termination_stats = {
            "landed": 0,
            "crash_high_speed": 0,
            "crash_bad_angle": 0,
            "out_of_bounds": 0,
            "time_limit": 0,
            "unknown": 0
        }

    def step(self, state, action, reward, next_state, done):
        self._nstep_buffer.append((state, action, reward, next_state, done))
        if len(self._nstep_buffer) == self.n_step:
            state_n, action_n, reward_n, next_state_n, done_n = self._get_nstep_info()
            self.memory.add(state_n, action_n, reward_n, next_state_n, done_n)

        if done:
            while len(self._nstep_buffer) > 0:
                state_n, action_n, reward_n, next_state_n, done_n = self._get_nstep_info()
                self.memory.add(state_n, action_n, reward_n, next_state_n, done_n)
                self._nstep_buffer.popleft()

        step_every = 4
        self.nb_step = (self.nb_step + 1) % step_every
        if self.nb_step == 0:
            if len(self.memory) > self.batch_size:
                experiences, indices, is_weights = self.memory.sample()
                self.learn(experiences, self.discount_factor_g, indices, is_weights)

    def _get_nstep_info(self):
        """Calcule le return N-step à partir du début du buffer."""
        state_n  = self._nstep_buffer[0][0]
        action_n = self._nstep_buffer[0][1]

        reward_n = 0.0
        for i, (_, _, r, _, d) in enumerate(self._nstep_buffer):
            reward_n += (self.discount_factor_g ** i) * r
            if d:
                break

        next_state_n = self._nstep_buffer[-1][3]
        done_n       = self._nstep_buffer[-1][4]
        return state_n, action_n, reward_n, next_state_n, done_n

    def act(self, state, fuel, eps=0.):
        state = torch.from_numpy(state).float().unsqueeze(0).to(device)
        self.qnetwork_local.eval()
        with torch.no_grad():
            dist = self.qnetwork_local(state)           # (1, actions, atoms)
            # ADD C51 — E[Q] = Σ z_i * p_i  pour chaque action
            expected_q = (dist * self.support).sum(dim=2)  # (1, actions)
        self.qnetwork_local.train()

        if self.enable_fuel and fuel <= 0:
            return 0 # do nothing if no more fuel
        # ADD C51
        return np.argmax(expected_q.cpu().data.numpy())

    # ADD C51
    def learn(self, experiences, gamma, indices, is_weights):
        states, actions, rewards, next_states, dones = experiences
        batch_size = states.shape[0]
        gamma_n = gamma ** self.n_step

        # ── Étape 1 : construire la distribution cible m ─────────────────────
        with torch.no_grad():
            # DDQN : le réseau local choisit la meilleure action …
            next_dist_local  = self.qnetwork_local(next_states)          # (B, A, Z)
            next_q_local     = (next_dist_local * self.support).sum(2)   # (B, A)
            best_actions     = next_q_local.argmax(1)                    # (B,)

            # … le réseau cible évalue la distribution de cette action
            next_dist_target = self.qnetwork_target(next_states)         # (B, A, Z)
            next_dist_target = next_dist_target[range(batch_size), best_actions]  # (B, Z)

            # Opérateur de Bellman distribitionnel : Tz = clip(r + γⁿ·z, Vmin, Vmax)
            Tz = rewards + (1 - dones) * gamma_n * self.support.unsqueeze(0)  # (B, Z)
            Tz = Tz.clamp(self.v_min, self.v_max)

            # Projection de Tz sur le support discret
            b = (Tz - self.v_min) / self.delta_z   # position continue dans le support
            l = b.floor().clamp(0, self.n_atoms - 1).long()
            u = b.ceil().clamp(0, self.n_atoms - 1).long()

            # Distribution cible m — redistribue la masse entre atomes voisins
            m = torch.zeros(batch_size, self.n_atoms, device=device)
            offset = (torch.arange(batch_size, device=device) * self.n_atoms
                    ).unsqueeze(1).expand_as(l)

            m.view(-1).index_add_(0, (l + offset).view(-1),
                                (next_dist_target * (u.float() - b)).view(-1))
            m.view(-1).index_add_(0, (u + offset).view(-1),
                                (next_dist_target * (b - l.float())).view(-1))

        # ── Étape 2 : distribution prédite pour les actions effectuées ────────
        dist     = self.qnetwork_local(states)                            # (B, A, Z)
        log_p    = torch.log(dist[range(batch_size), actions.squeeze(1)] + 1e-8)  # (B, Z)

        # ── Étape 3 : cross-entropie pondérée par les poids IS (PER) ─────────
        element_wise_loss = -(m * log_p).sum(dim=1, keepdim=True)        # (B, 1)
        loss = (is_weights * element_wise_loss).mean()

        # La loss cross-entropie sert directement de proxy pour les priorités PER
        td_errors = element_wise_loss.squeeze(1).detach().cpu().numpy()

        self.optimizer.zero_grad()
        loss.backward()
        clip_grad_norm_(self.qnetwork_local.parameters(), max_norm=10.0)
        self.optimizer.step()

        self.memory.update_priorities(indices, td_errors)
        self.soft_update(self.qnetwork_local, self.qnetwork_target, tau=0.001)
        self.qnetwork_local.reset_noise()
        self.qnetwork_target.reset_noise()              

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
        # ADD C51 — le support z = {V_min, ..., V_max} en 51 points réguliers
        self.support = torch.linspace(self.v_min, self.v_max, self.n_atoms).to(device)
        self.delta_z = (self.v_max - self.v_min) / (self.n_atoms - 1)

        self.qnetwork_local  = QNetwork(self.state_size, self.action_size, self.seed, self.activation_function,
                                        self.nb_neuron, self.nb_layer,
                                        n_atoms=self.n_atoms).to(device)  # ADD C51
        self.qnetwork_target = QNetwork(self.state_size, self.action_size, self.seed, self.activation_function,
                                        self.nb_neuron, self.nb_layer,
                                        n_atoms=self.n_atoms).to(device)  # ADD C51

        self.memory = PrioritizedReplayBuffer(action_size=self.action_size, state_size=self.state_size, buffer_size=self.replay_memory_size,
                alpha=self.per_alpha, beta_start=self.per_beta_start, beta_frames=self.per_beta_frames,
                batch_size=self.batch_size, seed=self.seed, device=device)
        self.optimizer = self.optimizer_function(self.qnetwork_local.parameters(), lr=self.lr) #TODO
        self._nstep_buffer = deque(maxlen=self.n_step)

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

        start_time = datetime.now()
        score = 0.0
        scores = []
        mean_rewards = []
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
        AgentRainbowDQN.plot_scores(scores, self.MODEL_FILE + "detail.png")
        return scores

# ─────────────────────────────────────────────────────────────────────────────
#  SumTree
#  A binary tree where each leaf stores a priority p_i and each internal node
#  stores the sum of its children.  This allows O(log n) sampling and updates.
#
#  Layout (capacity=4):
#         [0]          ← root  (sum of all priorities)
#        /    \
#      [1]    [2]
#      / \    / \
#    [3] [4] [5] [6]  ← leaves (actual priorities)
#
#  Leaf at position i  → tree index  i + capacity - 1
# ─────────────────────────────────────────────────────────────────────────────

class SumTree:
    def __init__(self, capacity: int):
        self.capacity = capacity
        self.tree     = np.zeros(2 * capacity - 1, dtype=np.float64)
        self.write    = 0          # next leaf to overwrite
        self.n_entries = 0         # filled entries (≤ capacity)

    # ── internal helpers ────────────────────────────────────────────────────

    def _leaf_idx(self, data_idx: int) -> int:
        return data_idx + self.capacity - 1

    def _propagate(self, leaf_idx: int, delta: float):
        """Bubble a change of `delta` up to the root."""
        parent = (leaf_idx - 1) // 2
        while True:
            self.tree[parent] += delta
            if parent == 0:
                break
            parent = (parent - 1) // 2

    # ── public API ──────────────────────────────────────────────────────────

    @property
    def total(self) -> float:
        return float(self.tree[0])

    def update(self, data_idx: int, priority: float):
        """Update the priority of experience at `data_idx`."""
        leaf = self._leaf_idx(data_idx)
        delta = priority - self.tree[leaf]
        self.tree[leaf] = priority
        self._propagate(leaf, delta)

    def add(self, priority: float):
        """
        Write a new priority at the current write head and advance it.
        Returns the data_idx used so the caller can store the experience.
        """
        data_idx = self.write
        self.update(data_idx, priority)
        self.write = (self.write + 1) % self.capacity
        self.n_entries = min(self.n_entries + 1, self.capacity)
        return data_idx

    def get(self, s: float):
        """
        Retrieve the leaf whose prefix sum interval contains `s`.
        Returns (data_idx, priority).
        """
        idx = 0
        while idx < self.capacity - 1:           # while not a leaf
            left = 2 * idx + 1
            if s <= self.tree[left]:
                idx = left
            else:
                s -= self.tree[left]
                idx = left + 1
        data_idx = idx - (self.capacity - 1)
        return data_idx, float(self.tree[idx])


# ─────────────────────────────────────────────────────────────────────────────
#  PrioritizedReplayBuffer
#
#  Hyper-parameters
#  ─────────────────
#  alpha (α) : priority exponent – controls how much prioritisation to use.
#              α = 0 → uniform sampling  |  α = 1 → full prioritisation
#  beta  (β) : IS-weight exponent – corrects the bias introduced by non-uniform
#              sampling.  β is annealed from beta_start → 1.0 over training.
#  eps       : small constant added to every TD error to ensure non-zero priority
#              and avoid starving rarely-seen transitions.
# ─────────────────────────────────────────────────────────────────────────────

class PrioritizedReplayBuffer:

    def __init__(
        self,
        action_size:  int,
        state_size,
        buffer_size:  int,
        batch_size:   int,
        seed:         int,
        alpha:        float = 0.6,
        beta_start:   float = 0.4,
        beta_frames:  int   = 100_000,
        eps:          float = 1e-6,
        device: str = "cpu",
    ):
        self.action_size = action_size
        self.batch_size  = batch_size
        self.alpha       = alpha
        self.beta_start  = beta_start
        self.beta_frames = beta_frames
        self.eps         = eps
        self.device      = device
        self.frame       = 1          # incremented each time sample() is called

        random.seed(seed)
        np.random.seed(seed)

        self.tree = SumTree(buffer_size)
        # Pre-allocate numpy arrays for O(1) index-access to stored experiences
        self.states      = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.next_states = np.zeros((buffer_size, state_size), dtype=np.float32)
        self.actions     = np.zeros((buffer_size,), dtype=np.int64)
        self.rewards     = np.zeros((buffer_size,), dtype=np.float32)
        self.dones       = np.zeros((buffer_size,), dtype=np.float32)

        self._max_priority = 1.0    # maintained so new transitions start high

    # ── helpers ─────────────────────────────────────────────────────────────

    def _beta(self) -> float:
        """Linear annealing of β from beta_start to 1.0."""
        progress = min(1.0, self.frame / self.beta_frames)
        return self.beta_start + progress * (1.0 - self.beta_start)

    def _priority(self, td_error: float) -> float:
        return (abs(td_error) + self.eps) ** self.alpha

    # ── public API ──────────────────────────────────────────────────────────

    def add(self, state, action, reward, next_state, done):
        """Add a new transition with maximum priority so it is sampled soon."""
        data_idx = self.tree.add(self._max_priority)
        self.states[data_idx]      = state
        self.actions[data_idx]     = action
        self.rewards[data_idx]     = reward
        self.next_states[data_idx] = next_state
        self.dones[data_idx]       = float(done)

    def sample(self):
        """
        Sample a mini-batch using stratified (segment) sampling.

        Returns
        -------
        states, actions, rewards, next_states, dones : torch.Tensor
        indices   : np.ndarray  – used to call update_priorities() after learn()
        is_weights: torch.Tensor – importance-sampling weights for the loss
        """
        beta = self._beta()
        self.frame += 1

        indices    = np.empty(self.batch_size, dtype=np.int32)
        priorities = np.empty(self.batch_size, dtype=np.float64)

        # Stratified sampling: divide the priority sum into equal segments
        segment = self.tree.total / self.batch_size
        for i in range(self.batch_size):
            lo = segment * i
            hi = segment * (i + 1)
            s  = random.uniform(lo, hi)
            idx, p = self.tree.get(s)
            indices[i]    = idx
            priorities[i] = p

        # Importance-sampling weights
        # w_i = (1/N · 1/P(i))^β   normalised by max weight for stability
        n          = self.tree.n_entries
        probs      = priorities / self.tree.total
        raw_weights = (n * probs) ** (-beta)
        is_weights  = (raw_weights / raw_weights.max()).astype(np.float32)

        # Build batched tensors
        states = torch.from_numpy(self.states[indices]).float().to(self.device)
        actions     = torch.from_numpy(self.actions[indices]).long().unsqueeze(1).to(self.device)
        rewards     = torch.from_numpy(self.rewards[indices]).float().unsqueeze(1).to(self.device)
        next_states = torch.from_numpy(np.vstack(self.next_states[indices])).float().to(self.device)
        dones       = torch.from_numpy(self.dones[indices]).float().unsqueeze(1).to(self.device)
        is_weights  = torch.from_numpy(is_weights).float().unsqueeze(1).to(self.device)

        return (states, actions, rewards, next_states, dones), indices, is_weights

    def update_priorities(self, indices: np.ndarray, td_errors: np.ndarray):
        """Called by the agent after each learning step."""
        for idx, err in zip(indices, td_errors):
            p = self._priority(float(err))
            self.tree.update(int(idx), p)
            self._max_priority = max(self._max_priority, p)

    def __len__(self) -> int:
        return self.tree.n_entries