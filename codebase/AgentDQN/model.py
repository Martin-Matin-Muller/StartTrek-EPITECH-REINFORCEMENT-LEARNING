import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, hidden_layer_neuron=64, nb_layer=1):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        
        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layer_neuron))

        for _ in range(nb_layer):
            self.layers.append(nn.Linear(hidden_layer_neuron, hidden_layer_neuron))

        self.output = nn.Linear(hidden_layer_neuron, action_size)

    def forward(self, state):
        for layer in self.layers:
            state = F.relu(layer(state))

        return self.output(state)