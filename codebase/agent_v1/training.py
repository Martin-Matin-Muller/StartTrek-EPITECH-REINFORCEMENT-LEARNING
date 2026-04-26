from agent_v1.log_variable import device
import torch
import random

def _training(self, env, is_training, episode, policy_dqn, memory,
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
        if is_training and (random.random() < epsilon or episode < 100):
            action = env.action_space.sample()
            action = torch.tensor(action, dtype=torch.int64, device=device)
        else:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(0)).squeeze().argmax()

        # ===================== STEP ENV =====================
        new_state, reward, terminated, _, _ = env.step(action.item())

        episode_reward += reward

        new_state = torch.tensor(new_state, dtype=torch.float, device=device)
        reward = torch.tensor(reward, dtype=torch.float, device=device)

        # ===================== MEMORY =====================
        if is_training:
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