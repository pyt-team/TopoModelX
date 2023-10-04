"""HyperGat Layer."""

import torch

from topomodelx.nn.hypergraph.hypergat_layer import HyperGATLayer


class HyperGAT(torch.nn.Module):
    """Neural network implementation of Template for hypergraph classification [1]_.

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
    .. [1] Ding, Wang, Li, Li and Huan Liu.
        EMNLP, 2020.
        https://aclanthology.org/2020.emnlp-main.399.pdf
    """

    def __init__(self, in_channels, out_channels, n_layers=2):
        super().__init__()
        layers = []
        layers.append(HyperGATLayer(in_channels=in_channels, out_channels=out_channels))
        for _ in range(1, n_layers):
            layers.append(
                HyperGATLayer(in_channels=out_channels, out_channels=out_channels)
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x_1, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_1 : torch.Tensor
            shape = (n_edges, channels_edge)
            Edge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_1 = layer.forward(x_1, incidence_1)
        pooled_x = torch.max(x_1, dim=0)[0]
        return torch.sigmoid(self.linear(pooled_x))[0]
