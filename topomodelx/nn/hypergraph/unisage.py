"""UniSAGE class."""

import torch

from topomodelx.nn.hypergraph.unisage_layer import UniSAGELayer


class UniSAGE(torch.nn.Module):
    """Neural network implementation of UniSAGE [1]_ for hypergraph classification.

    Parameters
    ----------
    channels_edge : int
        Dimension of edge features
    channels_node : int
        Dimension of node features
    n_layer : int, default = 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(self, channels_edge, channels_node, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                UniSAGELayer(
                    in_channels=channels_edge,
                    out_channels=channels_edge,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(channels_edge, 1)

    def forward(self, x_1, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (n_edges, channels_edge)
            Edge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_1 = layer(x_1, incidence_1)
        pooled_x = torch.max(x_1, dim=0)[0]
        return torch.sigmoid(self.linear(pooled_x))
