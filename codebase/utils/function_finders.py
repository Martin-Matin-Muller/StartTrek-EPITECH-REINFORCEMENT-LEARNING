import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def get_optimizer(name):
    return {
        "adam": optim.Adam,
        "sgd": optim.SGD,
        "rmsprop": optim.RMSprop,
        # add other opitmizer
    }[name.lower()]

def get_activation(name):
    return {
        "relu": nn.ReLU,
        "tanh": nn.Tanh,
        "leaky_relu": nn.LeakyReLU,
        # add other activation function
    }[name.lower()]