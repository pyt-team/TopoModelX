"""HNHN class."""

import torch

from topomodelx.nn.hypergraph.hnhn_layer import HNHNLayer


class HNHN(torch.nn.Module):
    """Hypergraph Networks with Hyperedge Neurons [1]_. Implementation for multiclass node classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    incidence_1 : torch.sparse, shape = (n_nodes, n_edges)
        Incidence matrix mapping edges to nodes (B_1).
    n_layers : int, default = 2
        Number of HNHN message passing layers.

    References
    ----------
    .. [1] Dong, Sawin, Bengio.
        HNHN: hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020.
        https://grlplus.github.io/papers/40.pdf
    """

    def __init__(self, in_channels, hidden_channels, incidence_1, n_layers=2):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            HNHNLayer(
                in_channels=in_channels if i == 0 else hidden_channels,
                hidden_channels=hidden_channels,
                incidence_1=incidence_1,
            )
            for i in range(n_layers)
        )

    def forward(self, x_0):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels_node)
            Hypernode features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0)

        return x_0, x_1
