import torch
import torch.nn as nn
import torch.nn.functional as F
import math

class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.sigma_init = sigma_init
        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        bound = 1.0 / math.sqrt(self.in_features)
        self.weight_mu.data.uniform_(-bound, bound)
        self.bias_mu.data.uniform_(-bound, bound)
        self.weight_sigma.data.fill_(self.sigma_init / math.sqrt(self.in_features))
        self.bias_sigma.data.fill_(self.sigma_init / math.sqrt(self.out_features))

    def reset_noise(self):
        """
        Factorised Gaussian noise (plus efficace que le bruit indépendant).
        Au lieu de tirer in*out valeurs, on tire in + out valeurs et on les combine.
        """
        eps_in  = self._f(torch.randn(self.in_features))
        eps_out = self._f(torch.randn(self.out_features))
        self.weight_epsilon.copy_(eps_out.outer(eps_in))
        self.bias_epsilon.copy_(eps_out)

    @staticmethod
    def _f(x):
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, activation, hidden_layer_neuron=64, nb_layer=1,
                        n_atoms=51): # ADD C51
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)
        self.n_atoms   = n_atoms                       # ADD C51
        self.action_size = action_size
        self.activation = activation

        self.layers = nn.ModuleList()
        self.layers.append(NoisyLinear(state_size, hidden_layer_neuron))
        for _ in range(nb_layer):
            self.layers.append(NoisyLinear(hidden_layer_neuron, hidden_layer_neuron))

        # ADD C51 — chaque action a n_atoms probabilités au lieu d'un scalaire
        self.value_stream     = NoisyLinear(hidden_layer_neuron, n_atoms)
        self.advantage_stream = NoisyLinear(hidden_layer_neuron, action_size * n_atoms)


    def forward(self, state):
        batch = state.shape[0]
        for layer in self.layers:
            state = self.activation(layer(state))

        # ADD C51 — Dueling sur les distributions
        V = self.value_stream(state).view(batch, 1, self.n_atoms)
        A = self.advantage_stream(state).view(batch, self.action_size, self.n_atoms)

        Q_atoms = V + (A - A.mean(dim=1, keepdim=True))  # (batch, actions, atoms)

        # Softmax sur les atomes → distribution de probabilités par action
        return F.softmax(Q_atoms, dim=2)                  # (batch, actions, atoms)

    def reset_noise(self):
        """Appelé par l'agent à chaque step pour renouveler le bruit."""
        for layer in self.layers:
            layer.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()