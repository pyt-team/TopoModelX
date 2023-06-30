# from typing import Callable

# import torch
# import torch.nn as nn
# import torch.nn.functional as F
# from torch import Tensor

# from topomodelx.base.message_passing import MessagePassing
# from topomodelx.utils.scatter import scatter_add

# class CAPooLayer(nn.Module):
#     """
#     CAPooLayer (Cellular Attention Pooling Layer) is responsible for pooling
#     operations in the Cellular Graph Attention Network.

#     This layer applies attention-based pooling to a given edge signal
#     and updates the graph accordingly.

#     Parameters
#     ----------
#     k_pool : float
#         Fraction of edges to keep after the pooling operation.
#     F_in : int
#         Number of input features for the pooling layer.
#     cell_forward_activation : Callable
#         Non-linear activation function used in the forward pass.

#     Returns
#     -------
#     CAPooLayer.

#     Examples
#     -------
#     pool = CAPooLayer(k_pool=.75,
#                       F_in=3*att_heads,
#                       cell_forward_activation=nn.ReLU)

#     References
#     ----------
#     .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
#         Cell attention networks. (2022)
#         paper: https://arxiv.org/pdf/2209.08179.pdf
#         repository: https://github.com/lrnzgiusti/can
#     """

#     def __init__(self, k_pool: float, F_in: int, cell_forward_activation: Callable):
#         super(CAPooLayer, self).__init__()

#         self.k_pool = k_pool
#         self.cell_forward_activation = cell_forward_activation

#         # Learnable attention parameter for the pooling operation
#         self.att_pool = nn.Parameter(torch.empty(size=(F_in, 1)))

#         # Initialize the attention parameter using Xavier initialization
#         nn.init.xavier_normal_(self.att_pool.data, gain=1.41)


#     def __repr__(self):
#        s = "PoolLayer(" + \
#            "K Pool="+str(self.k_pool)+ ")"
#        return s


#     def forward(self,  x: EdgeSignal) -> EdgeSignal:

#         x, G = x
#         shape = x.shape
#         Zp = x @ self.att_pool
#         idx = topk(Zp.view(-1), self.k_pool, G.edge_batch)
#         x = x[idx] * self.cell_forward_activation(Zp)[idx].view(-1, 1)
#         G.edge_batch = G.edge_batch[idx]
#         G.ros.append(readout(x, G.edge_batch, 'sum'))
#         G.connectivities['up'] = tuple(filter_adj(torch.stack(G.connectivities['up']), None, idx, shape[0])[0])
#         G.connectivities['do'] = tuple(filter_adj(torch.stack(G.connectivities['do']), None, idx, shape[0])[0])


#         return x, G

"""Attentional Pooling Layer adapted from the official implementation of the CeLL Attention Network (CAN)."""

from typing import Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor, topk

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter_add


class PoolLayer(MessagePassing):
    """Attentional Pooling Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

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
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.att_pool.data, gain=gain)

    def forward(self, x_0, lower_neighborhood, upper_neighborhood) -> Tensor:
        """Forward pass.

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
        # TODO should I overwrite the attention function in MessagePassing?
        # Compute the output edge signal by applying the activation function
        Zp = torch.einsum("nc,ce->ne", x_0, self.att_pool)
        # Apply top-k pooling to the edge signal
        _, top_indices = topk(Zp.view(-1), int(self.k_pool * Zp.size(0)))
        # Rescale the pooled signal
        Zp = self.signal_pool_activation(Zp)
        out = x_0[top_indices] * Zp[top_indices]

        # Readout operation
        if self.readout:
            # TODO double check this and also should this be in the aggregation function of MessagePassing?
            out = scatter_add(out, top_indices, dim=0, dim_size=x_0.size(0))[
                top_indices
            ]

        # Update lower and upper neighborhood matrices with the top-k pooled edges
        lower_neighborhood = lower_neighborhood[top_indices]
        lower_neighborhood = lower_neighborhood[:, top_indices]
        upper_neighborhood = upper_neighborhood[top_indices]
        upper_neighborhood = upper_neighborhood[:, top_indices]

        return out, lower_neighborhood, upper_neighborhood


# main with class PoolLayer

# if __name__ == "__main__":
#     # Parameters
#     k_pool = 0.75
#     in_channels_0 = 96
#     signal_pool_activation = nn.ReLU()

#     # Input
#     x_0 = torch.randn(38, in_channels_0)
#     lower_neighborhood = torch.randn(38, 38)
#     upper_neighborhood = torch.randn(38, 38)

#     # Instantiate the PoolLayer
#     pool_layer = PoolLayer(
#         k_pool=k_pool,
#         in_channels_0=in_channels_0,
#         signal_pool_activation=signal_pool_activation,
#         readout=True,
#     )

#     # Forward pass
#     out = pool_layer.forward(x_0, lower_neighborhood, upper_neighborhood)
#     print(out.shape)
#     print(out)
