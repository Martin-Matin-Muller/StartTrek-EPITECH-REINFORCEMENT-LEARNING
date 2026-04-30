from ..log_variable import device
import torch

# Optimize Function =======================================================================
def optimize(self, mini_batch, policy_dqn, target_dqn, weights, indices, memory):
    states, actions, new_states, rewards, terminations = zip(*mini_batch)

    states = torch.stack(states)
    actions = torch.stack(actions)
    new_states = torch.stack(new_states)
    rewards = torch.stack(rewards)
    terminations = torch.tensor(terminations).float().to(device)
    weights = torch.tensor(weights, dtype=torch.float32, device=device)

    discount_factor = 0.99
    with torch.no_grad():
        best_actions_from_policy = policy_dqn(new_states).argmax(dim=1)
        target_q = rewards + (1-terminations) * discount_factor * \
                                    target_dqn(new_states).gather(dim=1, index=best_actions_from_policy.unsqueeze(dim=1)).squeeze()

    current_q = policy_dqn(states).gather(dim=1, index=actions.unsqueeze(dim=1)).squeeze()

    td_errors = torch.abs(current_q - target_q).detach()
    loss = (weights * (current_q - target_q) ** 2).mean()
    
    # Optimize the model (backpropagation)
    self.optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_norm_(policy_dqn.parameters(), 1.0)
    self.optimizer.step()       # Update network parameters (weights and biases)

    td_errors_np = td_errors.cpu().numpy()
    memory.update_priorities(indices, td_errors_np)
#==========================================================================================