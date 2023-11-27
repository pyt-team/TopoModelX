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
    layer_drop: float, default = 0.2
        Dropout rate for the hidden features.

    References
    ----------
    .. [1] Dong, Sawin, Bengio.
        HNHN: hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020.
        https://grlplus.github.io/papers/40.pdf
    """

    def __init__(
        self, in_channels, hidden_channels, incidence_1, n_layers=2, layer_drop=0.2
    ):
        super().__init__()

        layers = []
        layers.append(
            HNHNLayer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                incidence_1=incidence_1,
            )
        )
        for _ in range(n_layers - 1):
            layers.append(
                HNHNLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    incidence_1=incidence_1,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.layer_drop = torch.nn.Dropout(layer_drop)

    def forward(self, x_0, incidence_1=None):
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
            x_0, x_1 = layer(x_0, incidence_1)
            x_0 = self.layer_drop(x_0)

        return x_0, x_1
