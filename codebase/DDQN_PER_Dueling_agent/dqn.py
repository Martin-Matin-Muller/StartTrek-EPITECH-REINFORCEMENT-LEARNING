from torch import nn
import torch.nn.functional as F

class DQN(nn.Module):

    def __init__(self, state_dim, number_layer, action_dim, hidden_dim=8):
        super(DQN, self).__init__()

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_dim, hidden_dim))

        for _ in range(number_layer - 1):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        self.fc_value = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, 1)

        self.fc_advantage = nn.Linear(hidden_dim, hidden_dim)
        self.advantage = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        for layer in self.layers:
            x = F.relu(layer(x))

        v = F.relu(self.fc_value(x))
        V = self.value(v)

        a = F.relu(self.fc_advantage(x))
        A = self.advantage(a)

        Q = V + (A - A.mean(dim=1, keepdim=True))

        return Q