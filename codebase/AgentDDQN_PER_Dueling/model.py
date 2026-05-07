import torch
import torch.nn as nn
import torch.nn.functional as F

class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, activation, hidden_layer_neuron=64, nb_layer=1):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(nn.Linear(state_size, hidden_layer_neuron))
        for _ in range(nb_layer):
            self.layers.append(nn.Linear(hidden_layer_neuron, hidden_layer_neuron))

        # ADD DUELING — deux têtes séparées
        # V(s) : scalaire, valeur de l'état
        self.value_stream = nn.Linear(hidden_layer_neuron, 1)
        # A(s,a) : vecteur, avantage de chaque action
        self.advantage_stream = nn.Linear(hidden_layer_neuron, action_size)

    def forward(self, state):
        for layer in self.layers:
            state = self.activation(layer(state))

        # ADD DUELING — deux têtes, une combinaison
        V = self.value_stream(state)           # (batch, 1)
        A = self.advantage_stream(state)       # (batch, action_size)

        # Q = V(s) + A(s,a) − mean(A(s,a))
        # keepdim=True pour que la soustraction broadcaste correctement
        Q = V + (A - A.mean(dim=1, keepdim=True))
        return Q