"""HyperSAGE Layer."""

import torch

from topomodelx.nn.hypergraph.hypersage_layer import HyperSAGELayer


class HyperSAGE(torch.nn.Module):
    """Neural network implementation of HyperSAGE [1]_ for hypergraph classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default = 2
        Amount of message passing layers.
    alpha : int, default = -1
        Max number of nodes in a neighborhood to consider. If -1 it considers all the nodes.
    **kwargs : optional
        Additional arguments for the inner layers.

    References
    ----------
    .. [1] Arya, Gupta, Rudinac and Worring.
        HyperSAGE: Generalizing inductive representation learning on hypergraphs (2020).
        https://arxiv.org/abs/2010.04558
    """

    def __init__(self, in_channels, hidden_channels, n_layers=2, alpha=-1, **kwargs):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            HyperSAGELayer(
                in_channels=in_channels if i == 0 else hidden_channels,
                out_channels=hidden_channels,
                alpha=alpha,
                **kwargs,
            )
            for i in range(n_layers)
        )

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, features_nodes)
            Edge features.
        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0 = layer.forward(x_0, incidence_1)

        return x_0
