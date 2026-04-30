from .log_variable import device
import torch
import random

def _training(self, env, episode, policy_dqn, target_dqn, memory,
              epsilon, action_counts, step_count):
    state, _ = env.reset()
    state = torch.tensor(state, dtype=torch.float, device=device)
    terminated = False
    episode_reward = 0.0
    max_steps = 1000
    steps = 0

    while not terminated and steps < max_steps:
        steps += 1

        # Action selection
        if random.random() < epsilon:
            action = torch.tensor(env.action_space.sample(), dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

        new_state, reward, terminated, _, _ = env.step(action.item())
        episode_reward += reward
        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward    = torch.tensor(reward,    dtype=torch.float, device=device)

        memory.append((state, action, new_state, reward, terminated))
        step_count += 1
        state = new_state
        action_counts[action.item()] += 1

        if len(memory) > 10000 and step_count % 4 == 0:
            mini_batch, indices, weights = memory.sample(self.mini_batch_size)
            self.optimize(mini_batch, policy_dqn, target_dqn, weights, indices, memory)

        if step_count % 1000 == 0:
            target_dqn.load_state_dict(policy_dqn.state_dict())

        if steps >= max_steps:
            terminated = True

    return episode_reward, step_count