from abc import ABC
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as f


class FullyConnectedNeuralNetwork(nn.Module, ABC):
    def __init__(self, input_dim: int, output_dim: int, hidden_sizes: List[int]):
        super(FullyConnectedNeuralNetwork, self).__init__()
        self.layer_sizes = [input_dim] + hidden_sizes + [output_dim]
        self.layer_dims = zip(self.layer_sizes[:-1], self.layer_sizes[1:])
        self.linear_layers = nn.ModuleList([
            nn.Linear(input_dim, output_dim) for input_dim, output_dim in self.layer_dims
        ])
        # self.batch_norm_layers = nn.ModuleList([
        #     nn.BatchNorm1d(dim) for dim in hidden_sizes
        # ])

    def forward(self, input_tensor: torch.Tensor):
        hidden_tensor = input_tensor
        for layer_num, layer in enumerate(self.linear_layers):
            hidden_tensor = layer(hidden_tensor)
            if layer_num < len(self.linear_layers) - 1:
                # hidden_tensor = self.batch_norm_layers[layer_num](hidden_tensor)
                hidden_tensor = f.leaky_relu(hidden_tensor)

        return hidden_tensor


class TrainedVAE(object):
    def __init__(self, encoder, decoder, property_network, scaler_list, scaler_key):
        self.encoder = encoder
        self.decoder = decoder
        self.property_network = property_network
        self.scaler_list = scaler_list
