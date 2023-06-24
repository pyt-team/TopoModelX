import torch
import torch.nn as nn
from torch import Tensor
import torch.nn.functional as F

from typing import Callable

"""
Attentional Lift Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

References
----------
[CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
    paper: https://arxiv.org/pdf/2209.08179.pdf
    repository: https://github.com/lrnzgiusti/can

"""


class LiftLayer(nn.Module):

    def __init__(self, in_channels_0: int,
                 signal_lift_activation: Callable,
                 signal_lift_dropout: float):
        super(LiftLayer, self).__init__()

        self.in_channels_0 = in_channels_0
        self.att = nn.Parameter(torch.empty(size=(2 * in_channels_0, 1))).to(torch.cuda.current_device())
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = nn.init.calculate_gain('relu')
        nn.init.xavier_uniform_(self.att.data, gain=gain)

    def forward(self, x_0, neighborhood_0_to_0) -> Tensor:

        # Extract source and target nodes from the graph's edge index
        source, target = neighborhood_0_to_0.indices()

        # Concatenate source and target node feature vectors
        node_features_stacked = torch.cat((x_0[source], x_0[target]), dim=1)

        # Compute the output edge signal by applying the activation function
        edge_signal = self.signal_lift_activation(node_features_stacked.mm(self.att))

        return edge_signal


class MultiHeadLiftLayer(nn.Module):

    def __init__(self, in_channels_0: int, K: int = 3,
                 signal_lift_activation: Callable = torch.relu,
                 signal_lift_dropout: float = 0.0,
                 signal_lift_readout: str = 'cat', *args, **kwargs):
        super(MultiHeadLiftLayer, self).__init__()

        self.in_channels_0 = in_channels_0
        self.K = K
        self.signal_lift_readout = signal_lift_readout
        self.signal_lift_dropout = signal_lift_dropout
        self.signal_lift_activation = signal_lift_activation
        self.lifts = [LiftLayer(in_channels_0=in_channels_0,
                                signal_lift_activation=signal_lift_activation,
                                signal_lift_dropout=signal_lift_dropout) for _ in range(K)]


    def forward(self, x_0, x_1, neighborhood_0_to_0) -> Tensor:

        # Lift the node signal for each attention head
        attention_heads_x_1 = [lift(x_0, neighborhood_0_to_0) for lift in self.lifts]

        # Combine the output edge signals using the specified readout strategy
        if self.signal_lift_readout == 'cat':
            combined_x_1 = torch.cat(attention_heads_x_1, dim=1)
        elif self.signal_lift_readout == 'sum':
            combined_x_1 = torch.stack(attention_heads_x_1, dim=2).sum(dim=2)
        elif self.signal_lift_readout == 'avg':
            combined_x_1 = torch.stack(attention_heads_x_1, dim=2).mean(dim=2)
        elif self.signal_lift_readout == 'max':
            combined_x_1 = torch.stack(attention_heads_x_1, dim=2).max(dim=2).values
        else:
            raise ValueError("Invalid signal_lift_readout value. Choose from ['cat', 'sum', 'avg', 'max']")

        # Apply dropout to the combined edge signal
        combined_x_1 = F.dropout(combined_x_1, self.signal_lift_dropout, training=self.training)

        # Concatenate the lifted node signal with the original node signal if is not None
        if x_1 is not None:
            combined_x_1 = torch.cat((combined_x_1, x_1), dim=1)

        return combined_x_1