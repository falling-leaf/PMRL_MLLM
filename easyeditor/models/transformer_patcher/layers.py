"""Paired FFN expansion layers used by Transformer-Patcher.

The original FFN is frozen.  New output channels in fc1 and matching new input
columns in fc2 form an additive, trainable neuron path while preserving the
original computation exactly at initialization.
"""

import copy

import torch
from torch import nn
from torch.nn import functional as F


class ExpandedLinearOutput(nn.Module):
    """Expand a linear layer's output dimension with trainable neurons."""

    def __init__(self, linear: nn.Linear, add_neuron_num: int = 1):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("Transformer-Patcher requires nn.Linear fc1/fc2 modules")
        if add_neuron_num < 1:
            raise ValueError("add_neuron_num must be positive")
        self.linear = copy.deepcopy(linear)
        self.add_neuron_num = int(add_neuron_num)
        for parameter in self.linear.parameters():
            parameter.requires_grad_(False)
        self.extra_output = nn.Linear(
            linear.in_features, self.add_neuron_num, bias=linear.bias is not None
        ).to(device=linear.weight.device, dtype=linear.weight.dtype)
        nn.init.kaiming_uniform_(self.extra_output.weight, a=5**0.5)
        if self.extra_output.bias is not None:
            nn.init.zeros_(self.extra_output.bias)

    def forward(self, hidden_states):
        return torch.cat((self.linear(hidden_states), self.extra_output(hidden_states)), dim=-1)


class ExpandedLinearInput(nn.Module):
    """Consume a paired fc1 expansion and add its contribution to fc2."""

    def __init__(self, linear: nn.Linear, add_neuron_num: int = 1):
        super().__init__()
        if not isinstance(linear, nn.Linear):
            raise TypeError("Transformer-Patcher requires nn.Linear fc1/fc2 modules")
        if add_neuron_num < 1:
            raise ValueError("add_neuron_num must be positive")
        self.linear = copy.deepcopy(linear)
        self.add_neuron_num = int(add_neuron_num)
        for parameter in self.linear.parameters():
            parameter.requires_grad_(False)
        self.extra_input = nn.Linear(self.add_neuron_num, linear.out_features, bias=False).to(
            device=linear.weight.device, dtype=linear.weight.dtype
        )
        nn.init.zeros_(self.extra_input.weight)

    def forward(self, hidden_states):
        return self.linear(hidden_states[..., :-self.add_neuron_num]) + self.extra_input(
            hidden_states[..., -self.add_neuron_num:]
        )
