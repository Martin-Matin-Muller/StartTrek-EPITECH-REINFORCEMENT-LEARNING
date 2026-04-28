from ..log_variable import DATE_FORMAT, device
from datetime import datetime
import torch

from ..experience_replay import ReplayMemory, PrioritizedReplayMemory
from .dqn import DQN

#========================================================

def _setup_logging(self, is_training):
    if is_training:
        start_time = datetime.now()
        last_graph_update_time = start_time

        log_message = f"{start_time.strftime(DATE_FORMAT)}: Training starting..."
        print(log_message)
        with open(self.LOG_FILE, 'w') as file:
            file.write(log_message + '\n')
        return last_graph_update_time, log_message
    return None, None

def _setup_training(self, is_training, env):
    num_actions = env.action_space.n
    num_states = env.observation_space.shape[0]
    policy_dqn = DQN(num_states, self.number_of_layer, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)

    if is_training:
        epsilon = self.epsilon_init

        memory = ReplayMemory(self.replay_memory_size, self.seed) 
        # memory = PrioritizedReplayMemory(self.replay_memory_size)
        epsilon_history = []
        step_count=0
        best_reward = -9999999

        # Create the target network and make it identical to the policy network
        target_dqn = DQN(num_states, self.number_of_layer, num_actions, self.fc1_nodes, self.enable_dueling_dqn).to(device)
        target_dqn.load_state_dict(policy_dqn.state_dict())

        # Policy network optimizer. "Adam" optimizer can be swapped to something else.
        self.optimizer = torch.optim.Adam(policy_dqn.parameters(), lr=self.learning_rate_a)

        return policy_dqn, target_dqn, memory, epsilon, epsilon_history, step_count, best_reward
    else:
        # policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        # policy_dqn.eval()

        checkpoint = torch.load(self.MODEL_FILE, map_location=device)
        state_dict = checkpoint.get("model_state_dict", checkpoint)
        policy_dqn.load_state_dict(state_dict)
        policy_dqn.to(device)
        policy_dqn.eval()
        return policy_dqn, None, None, None, None, None, None