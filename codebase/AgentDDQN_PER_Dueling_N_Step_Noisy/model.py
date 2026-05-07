import torch
import torch.nn as nn
import torch.nn.functional as F
import math

# Add Noisy
class NoisyLinear(nn.Module):
    def __init__(self, in_features, out_features, sigma_init=0.5):
        super(NoisyLinear, self).__init__()
        self.in_features  = in_features
        self.out_features = out_features

        # Paramètres appris — μ (valeur centrale) et σ (amplitude du bruit)
        self.weight_mu    = nn.Parameter(torch.empty(out_features, in_features))
        self.weight_sigma = nn.Parameter(torch.empty(out_features, in_features))
        self.bias_mu      = nn.Parameter(torch.empty(out_features))
        self.bias_sigma   = nn.Parameter(torch.empty(out_features))

        # Bruit non appris, mis à jour à chaque reset_noise()
        self.register_buffer("weight_epsilon", torch.empty(out_features, in_features))
        self.register_buffer("bias_epsilon",   torch.empty(out_features))

        self.sigma_init = sigma_init
        self._init_parameters()
        self.reset_noise()

    def _init_parameters(self):
        # Initialisation recommandée par le papier NoisyNets (Fortunato et al.)
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
        # Transformation recommandée par le papier pour le bruit factorisé
        return x.sign() * x.abs().sqrt()

    def forward(self, x):
        if self.training:
            # Pendant l'entraînement : poids bruités
            weight = self.weight_mu + self.weight_sigma * self.weight_epsilon
            bias   = self.bias_mu   + self.bias_sigma   * self.bias_epsilon
        else:
            # Pendant l'évaluation (act avec eval()) : poids moyens, pas de bruit
            weight = self.weight_mu
            bias   = self.bias_mu
        return F.linear(x, weight, bias)


class QNetwork(nn.Module):
    def __init__(self, state_size, action_size, seed, activation, hidden_layer_neuron=64, nb_layer=1):
        super(QNetwork, self).__init__()
        self.seed = torch.manual_seed(seed)

        self.activation = activation

        # Add Noisy Couches partagées — NoisyLinear à la place de nn.Linear
        self.layers = nn.ModuleList()
        self.layers.append(NoisyLinear(state_size, hidden_layer_neuron))
        for _ in range(nb_layer):
            self.layers.append(NoisyLinear(hidden_layer_neuron, hidden_layer_neuron))

        # Add Noisy Têtes Dueling — aussi en NoisyLinear
        self.value_stream     = NoisyLinear(hidden_layer_neuron, 1)
        self.advantage_stream = NoisyLinear(hidden_layer_neuron, action_size)

    def forward(self, state):
        for layer in self.layers:
            state = self.activation(layer(state))
        V = self.value_stream(state)
        A = self.advantage_stream(state)
        return V + (A - A.mean(dim=1, keepdim=True))

    # Add Noisy 
    def reset_noise(self):
        """Appelé par l'agent à chaque step pour renouveler le bruit."""
        for layer in self.layers:
            layer.reset_noise()
        self.value_stream.reset_noise()
        self.advantage_stream.reset_noise()