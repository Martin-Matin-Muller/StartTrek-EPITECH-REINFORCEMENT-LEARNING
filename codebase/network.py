import os
import yaml
import random
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

# class Network(nn.Module):
#   def __init__(self, state_size, action_size, seed=42):
#     super(Network, self).__init__()
#     self.seed=torch.manual_seed(seed)
#     self.fc1=nn.Linear(state_size,64)
#     self.fc2=nn.Linear(64,64)
#     self.fc3=nn.Linear(64,action_size)

#   def forward(self,state):
#     x=self.fc1(state)
#     x=F.relu(x)
#     x=self.fc2(x)
#     x=F.relu(x)
#     return self.fc3(x)

class DQN(nn.Module):

    def __init__(self, state_dim, action_dim, hidden_dim=256, enable_dueling_dqn=True):
        super(DQN, self).__init__()

        self.enable_dueling_dqn=enable_dueling_dqn

        self.fc1 = nn.Linear(state_dim, hidden_dim)

        if self.enable_dueling_dqn:
            # Value stream
            self.fc_value = nn.Linear(hidden_dim, 256)
            self.value = nn.Linear(256, 1)

            # Advantages stream
            self.fc_advantages = nn.Linear(hidden_dim, 256)
            self.advantages = nn.Linear(256, action_dim)

        else:
            self.output = nn.Linear(hidden_dim, action_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))

        if self.enable_dueling_dqn:
            # Value calc
            v = F.relu(self.fc_value(x))
            V = self.value(v)

            # Advantages calc
            a = F.relu(self.fc_advantages(x))
            A = self.advantages(a)

            # Calc Q
            Q = V + A - torch.mean(A, dim=1, keepdim=True)

        else:
            Q = self.output(x)

        return Q


if __name__ == '__main__':
    state_dim = 8
    action_dim = 4
    net = DQN(state_dim, action_dim)
    state = torch.randn(10, state_dim)
    output = net(state)
    print(output)