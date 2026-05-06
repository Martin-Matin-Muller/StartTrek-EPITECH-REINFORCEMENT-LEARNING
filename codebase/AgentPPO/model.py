import os

import torch as T
import torch.nn as nn
import torch.optim as optim
from torch.distributions.categorical import Categorical

from ..log_variable import device

class ActorNetwork(nn.Module):
    def __init__(self, n_actions, input_dims, alpha, activation, optimizer, hidden_dim, nb_layer):
        super(ActorNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(input_dims, hidden_dim))

        # Hidden layers
        for _ in range(nb_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, n_actions)

        # self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = T.relu(layer(x))
            # x = self.activation(layer(x))

        x = self.output(x)
        dist = Categorical(T.softmax(x, dim=-1))

        return dist


class CriticNetwork(nn.Module):
    def __init__(self, input_dims, alpha, activation, optimizer, hidden_dim, nb_layer):
        super(CriticNetwork, self).__init__()

        self.layers = nn.ModuleList()
        self.activation = activation

        # Input layer
        self.layers.append(nn.Linear(input_dims, hidden_dim))

        # Hidden layers
        for _ in range(nb_layer):
            self.layers.append(nn.Linear(hidden_dim, hidden_dim))

        # Output layer
        self.output = nn.Linear(hidden_dim, 1)

        # self.optimizer = optimizer(self.parameters(), lr=alpha)
        self.optimizer = optim.Adam(self.parameters(), lr=alpha)
        self.device = device
        self.to(self.device)

    def forward(self, state):
        x = state
        for layer in self.layers:
            x = T.relu(layer(x))
            # x = self.activation(layer(x))

        value = self.output(x)
        return value