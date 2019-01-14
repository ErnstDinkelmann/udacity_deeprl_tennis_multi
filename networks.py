# coding: utf-8
# Author: Ernst Dinkelmann

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return -lim, lim


class ActorNetwork(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_size, use_batch_norm, random_seed, fc1_units=512, fc2_units=512, fc3_units=512):
        """
        Initialize parameters and layers.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            use_batch_norm (bool): True to use batch normalisation
            random_seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(ActorNetwork, self).__init__()
        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)
            self.bn2 = nn.BatchNorm1d(fc1_units)
            self.bn3 = nn.BatchNorm1d(fc2_units)
            self.bn4 = nn.BatchNorm1d(fc3_units)

        # batch norm has bias included, disable linear layer bias
        self.use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(state_size, fc1_units, bias=self.use_bias)
        self.fc2 = nn.Linear(fc1_units, fc2_units, bias=self.use_bias)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=self.use_bias)
        self.fc4 = nn.Linear(fc3_units, action_size, bias=self.use_bias)
        self.reset_parameters()

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        if self.use_batch_norm:
            x = F.leaky_relu(self.fc1(self.bn1(state)))
            x = F.leaky_relu(self.fc2(self.bn2(x)))
            x = F.leaky_relu(self.fc3(self.bn3(x)))
            return torch.tanh(self.fc4(self.bn4(x)))
        else:
            x = F.leaky_relu(self.fc1(state))
            x = F.leaky_relu(self.fc2(x))
            x = F.leaky_relu(self.fc3(x))
            x = torch.tanh(self.fc4(x))
            return x

    def reset_parameters(self):
        """Specify ranges or values when parameters are reset at initialization"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        # if self.use_bias:
        #     self.fc1.bias.data.fill_(-0.1)
        #     self.fc2.bias.data.fill_(-0.1)
        #     self.fc3.bias.data.fill_(-0.1)
        #     self.fc4.bias.data.fill_(-0.1)


class CriticNetwork(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, use_batch_norm, random_seed, fc1_units=512, fc2_units=512, fc3_units=512):
        """
        Initialize parameters and layers.

        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            use_batch_norm (bool): True to use batch normalisation
            random_seed (int): Random seed
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
            fc3_units (int): Number of nodes in the third hidden layer
        """
        super(CriticNetwork, self).__init__()

        if random_seed is not None:
            np.random.seed(random_seed)
            torch.manual_seed(random_seed)

        if use_batch_norm:
            self.bn1 = nn.BatchNorm1d(state_size)

        # batch norm has bias included, disable linear layer bias
        self.use_bias = not use_batch_norm

        self.use_batch_norm = use_batch_norm
        self.fc1 = nn.Linear(state_size, fc1_units, bias=self.use_bias)
        self.fc2 = nn.Linear(fc1_units + action_size, fc2_units, bias=True)
        self.fc3 = nn.Linear(fc2_units, fc3_units, bias=True)
        self.fc4 = nn.Linear(fc3_units, 1, bias=True)
        self.reset_parameters()

    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        if self.use_batch_norm:
            x = F.leaky_relu(self.fc1(self.bn1(state)))
        else:
            x = F.leaky_relu(self.fc1(state))

        x = torch.cat((x, action), dim=1)
        x = F.leaky_relu(self.fc2(x))
        x = F.leaky_relu(self.fc3(x))
        x = self.fc4(x)
        return x

    def reset_parameters(self):
        """Specify ranges or values when parameters are reset at initialization"""
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(*hidden_init(self.fc3))
        self.fc4.weight.data.uniform_(-3e-3, 3e-3)
        # if self.use_bias:
        #     self.fc1.bias.data.fill_(-0.1)
        #     self.fc2.bias.data.fill_(-0.1)
        #     self.fc3.bias.data.fill_(-0.1)
        #     self.fc4.bias.data.fill_(-0.1)
