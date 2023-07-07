"""Attentional Pooling Layer adapted from the official implementation of the CeLL Attention Network (CAN)."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, topk
from torch.nn import init

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter_add


class PoolLayer(MessagePassing):
    r"""Attentional Pooling Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

    Parameters
    ----------
    k_pool: float
        The pooling ratio i.e, the fraction of edges to keep after the pooling operation. (0,1]
    in_channels_0: int
        Number of input channels of the input signal.
    signal_pool_activation: Callable
        Activation function applied to the pooled signal.
    readout: bool
        Whether to apply a readout operation to the pooled signal.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        k_pool: float,
        in_channels_0: int,
        signal_pool_activation: Callable,
        readout: True,
    ):
        super(PoolLayer, self).__init__()

        self.k_pool = k_pool
        self.in_channels_0 = in_channels_0
        self.readout = readout
        # Learnable attention parameter for the pooling operation
        self.att_pool = nn.Parameter(torch.empty(size=(in_channels_0, 1)))
        self.signal_pool_activation = signal_pool_activation

        # Initialize the attention parameter using Xavier initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = init.calculate_gain("relu")
        init.xavier_uniform_(self.att_pool.data, gain=gain)

    def forward(self, x_0, lower_neighborhood, upper_neighborhood) -> Tensor:
        r"""Forward pass.

        .. math::
            \begin{align*}
            &🟥 \quad m_{x}^{(r)} 
                = \gamma^t(h_x^t) = \tau^t (a^t\cdot h_x^t)\\
            &🟦 \quad h_x^{t+1,(r)} 
                = \phi^t(h_x^t, m_{x}^{(r)}), \forall x\in \mathcal C_r^{t+1}
            \end{align*}

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0).
        neighborhood_0_to_0: torch.Tensor
            Neighborhood matrix of shape (num_edges, 2).

        Returns
        -------
        out: torch.Tensor
            Pooled node signal of shape (num_pooled_nodes, in_channels_0).
        """
        # Compute the output edge signal by applying the activation function
        Zp = torch.einsum("nc,ce->ne", x_0, self.att_pool)
        # Apply top-k pooling to the edge signal
        _, top_indices = topk(Zp.view(-1), int(self.k_pool * Zp.size(0)))
        # Rescale the pooled signal
        Zp = self.signal_pool_activation(Zp)
        out = x_0[top_indices] * Zp[top_indices]

        # Readout operation
        if self.readout:
            out = scatter_add(out, top_indices, dim=0, dim_size=x_0.size(0))[
                top_indices
            ]

        # Update lower and upper neighborhood matrices with the top-k pooled edges
        lower_neighborhood_modified = torch.index_select(
            lower_neighborhood, 0, top_indices
        )
        lower_neighborhood_modified = torch.index_select(
            lower_neighborhood_modified, 1, top_indices
        )
        upper_neighborhood_modified = torch.index_select(
            upper_neighborhood, 0, top_indices
        )
        upper_neighborhood_modified = torch.index_select(
            upper_neighborhood_modified, 1, top_indices
        )

        return out
