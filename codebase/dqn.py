import torch
from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn = enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)

        if self.enable_dueling_dqn:

            # Value stream
            self.fc_value = nn.Linear(hidden_dim, hidden_dim)
            self.value = nn.Linear(hidden_dim, 1)

            # Advantage stream
            self.fc_advantage = nn.Linear(hidden_dim, hidden_dim)
            self.advantage = nn.Linear(hidden_dim, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))

        if self.enable_dueling_dqn:

            v = F.relu(self.fc_value(x))
            V = self.value(v)

            a = F.relu(self.fc_advantage(x))
            A = self.advantage(a)

            A_mean = A.mean(dim=1, keepdim=True)
            Q = V + (A - A_mean)

        else:
            Q = self.output(x)

        return Q