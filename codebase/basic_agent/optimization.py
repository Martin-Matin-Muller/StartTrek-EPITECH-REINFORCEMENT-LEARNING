from ..log_variable import device
import torch

# Optimize Function =======================================================================
def optimize(self, mini_batch, policy_dqn, target_dqn, weights, indices, memory):
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    # Stack tensors to create batch tensors
    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations).float().to(device)

    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    with torch.no_grad():
        if self.enable_double_dqn:
            best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
            target_q = rewards + (1-terminations) * 0.99 * \
                            target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()
        else:
            # Calculate target Q values (expected returns)
            target_q = rewards + (1-terminations) * 0.99 * target_dqn(new_states).max(dim=1)[0]

    # Calcuate Q values from current policy
    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    # Compute loss
    # loss = self.loss_fn(current_q, target_q)
    td_errors = torch.abs(current_q - target_q).detach()
    loss = (weights * (current_q - target_q) ** 2).mean()

    # Optimize the model (backpropagation)
    self.optimizer.zero_grad()
    loss.backward()
    self.optimizer.step()       # Update network parameters i.e. weights and biases

    td_errors_np = td_errors.cpu().numpy()
    memory.update_priorities(indices, td_errors_np)
#==========================================================================================
