"""Attentional Lift Layer adapted from the official implementation of the CeLL Attention Network (CAN)."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor

from topomodelx.base.message_passing import MessagePassing


class LiftLayer(MessagePassing):
    """Attentional Lift Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

    Parameters
    ----------
    in_channels_0: int
        Number of input channels of the node signal.
    signal_lift_activation: Callable
        Activation function applied to the lifted signal.
    signal_lift_dropout: float
        Dropout rate applied to the lifted signal.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int,
        signal_lift_activation: Callable,
        signal_lift_dropout: float,
    ):
        super(LiftLayer, self).__init__()

        self.in_channels_0 = in_channels_0
        self.att = nn.Parameter(torch.empty(size=(2 * in_channels_0, heads)))
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.att.data, gain=gain)

    def message(self, x_source, x_target=None):
        """Construct message from source 0-cells to target 1-cell."""
        # Concatenate source and target node feature vectors
        node_features_stacked = torch.cat(
            (x_source, x_target), dim=1
        )  # (num_edges, 2 * in_channels_0)

        # Compute the output edge signal by applying the activation function
        edge_signal = torch.einsum(
            "ij,jh->ih", node_features_stacked, self.att
        )  # (num_edges, heads)
        edge_signal = self.signal_lift_activation(edge_signal)  # (num_edges, heads)

        return edge_signal  # (num_edges, heads)

    def forward(self, x_0, neighborhood_0_to_0) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0)
        neighborhood_0_to_0: torch.Tensor
            Sparse neighborhood matrix of shape (num_nodes, num_nodes)

        Returns
        -------
        _: torch.Tensor
            Edge signal of shape (num_edges, 1)
        """
        # Extract source and target nodes from the graph's edge index
        source, target = neighborhood_0_to_0.indices()  # (num_edges,)

        # Extract the node signal of the source and target nodes
        x_source = x_0[source]  # (num_edges, in_channels_0)
        x_target = x_0[target]  # (num_edges, in_channels_0)

        # Compute the edge signal
        return self.message(x_source, x_target)  # (num_edges, 1)


class MultiHeadLiftLayer(nn.Module):
    r"""Multi Head Attentional Lift Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

    .. math::
        \begin{align*}
        &🟥 \quad m_{(y,z) \rightarrow x}^{(0 \rightarrow 1)}                &=&\ \alpha(h_y, h_z)\\
        &&=&\  \Theta(h_z||h_y)\\
        &🟦 \quad h_x^{(1)}                &=&\ \phi(h_x, m_x^{(1)})\\
        \end{align*}

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can

    Parameters
    ----------
    in_channels_0: int
        Number of input channels.
    K: int
        Number of attention heads.
    signal_lift_activation: Callable
        Activation function to apply to the output edge signal.
    signal_lift_dropout: float
        Dropout rate to apply to the output edge signal.
    signal_lift_readout: str
        Readout method to apply to the output edge signal.
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int = 3,
        signal_lift_activation: Callable = torch.relu,
        signal_lift_dropout: float = 0.0,
        signal_lift_readout: str = "cat",
        *args,
        **kwargs,
    ):
        super(MultiHeadLiftLayer, self).__init__()

        assert heads > 0, "Number of attention heads must be greater than 0."
        assert signal_lift_readout in [
            "cat",
            "sum",
            "avg",
            "max",
        ], "Invalid readout method."

        self.in_channels_0 = in_channels_0
        self.heads = heads
        self.signal_lift_readout = signal_lift_readout
        self.signal_lift_dropout = signal_lift_dropout
        self.signal_lift_activation = signal_lift_activation
        self.lifts = LiftLayer(
            in_channels_0=in_channels_0,
            heads=heads,
            signal_lift_activation=signal_lift_activation,
            signal_lift_dropout=signal_lift_dropout,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        self.lifts.reset_parameters()

    def forward(self, x_0, neighborhood_0_to_0, x_1=None) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0)
        neighborhood_0_to_0: torch.Tensor
            Edge index of shape (2, num_edges)
        x_1: torch.Tensor, optional
            Node signal of shape (num_edges, in_channels_1)

        Returns
        -------
        _: torch.Tensor
            Lifted node signal of shape (num_edges, heads + in_channels_1)
        """
        # Lift the node signal for each attention head
        attention_heads_x_1 = self.lifts(x_0, neighborhood_0_to_0)

        # Combine the output edge signals using the specified readout strategy
        if self.signal_lift_readout == "cat":
            combined_x_1 = attention_heads_x_1
        if self.signal_lift_readout == "sum":
            combined_x_1 = attention_heads_x_1.sum(dim=1)[:, None]  # (num_edges, 1)
        if self.signal_lift_readout == "avg":
            combined_x_1 = attention_heads_x_1.mean(dim=1)[:, None]  # (num_edges, 1)
        if self.signal_lift_readout == "max":
            combined_x_1 = attention_heads_x_1.max(dim=1).values[
                :, None
            ]  # (num_edges, 1)

        # Apply dropout to the combined edge signal
        combined_x_1 = F.dropout(
            combined_x_1, self.signal_lift_dropout, training=self.training
        )

        # Concatenate the lifted node signal with the original node signal if is not None
        if x_1 is not None:
            combined_x_1 = torch.cat(
                (combined_x_1, x_1), dim=1
            )  # (num_edges, heads + in_channels_1)

        return combined_x_1
