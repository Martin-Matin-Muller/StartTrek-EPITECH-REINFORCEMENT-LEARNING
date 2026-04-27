from datetime import datetime, timedelta
import torch

def _evaluation(self, is_training, episode, episode_reward, best_reward, rewards_per_episode,
                      epsilon_history, memory, policy_dqn, target_dqn, epsilon, step_count,
                      last_graph_update_time, log_message):
    pass
    # if not is_training:
    #     return best_reward, epsilon, step_count, last_graph_update_time


    # # ===================== GRAPH UPDATE =====================
    # current_time = datetime.now()
    # if current_time - last_graph_update_time > timedelta(seconds=60):
    #     self.save_graph(rewards_per_episode, epsilon_history)
    #     last_graph_update_time = current_time

    # # ===================== TRAINING STEP =====================
    # if len(memory) > self.mini_batch_size:
    #     # mini_batch = memory.sample(self.mini_batch_size)
    #     mini_batch, indices, weights = memory.sample(self.mini_batch_size)
    #     self.optimize(mini_batch, policy_dqn, target_dqn, weights, indices, memory)

    #     epsilon = max(epsilon * self.epsilon_decay, self.epsilon_min)
    #     epsilon_history.append(epsilon)

    #     step_count += 1

    #     if step_count > self.network_sync_rate:
    #         target_dqn.load_state_dict(policy_dqn.state_dict())
    #         step_count = 0

    # return best_reward, epsilon, step_count, last_graph_update_time