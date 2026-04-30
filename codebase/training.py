from .log_variable import device
import torch
import random

def _training(self, env, episode, policy_dqn, target_dqn, memory,
                    epsilon, action_counts, step_count, beta):

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
        if random.random() < epsilon:
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

        epsilon = max(epsilon - ((self.epsilon_init - self.epsilon_min) / self.epsilon_decay), self.epsilon_min)
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

        # ===================== OPTIMIZATION =====================
        if len(memory) >= self.mini_batch_size:
            mini_batch, indices, weights = memory.sample(self.mini_batch_size, beta=beta)
            self.optimize(mini_batch, policy_dqn, target_dqn, weights, indices, memory)

        if steps >= max_steps:
            terminated = True

    return episode_reward, step_count, epsilon