"""HyperGat Layer."""

import torch

from topomodelx.nn.hypergraph.hypergat_layer import HyperGATLayer


class HyperGAT(torch.nn.Module):
    """Neural network implementation of Template for hypergraph classification [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default = 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Ding, Wang, Li, Li and Huan Liu.
        EMNLP, 2020.
        https://aclanthology.org/2020.emnlp-main.399.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
    ):
        super().__init__()
        layers = []
        layers.append(
            HyperGATLayer(in_channels=in_channels, hidden_channels=hidden_channels)
        )
        for _ in range(1, n_layers):
            layers.append(
                HyperGATLayer(
                    in_channels=hidden_channels, hidden_channels=hidden_channels
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (n_edges, channels_edge)
            Edge features.
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
            x_0, x_1 = layer.forward(x_0, incidence_1)

        return x_0, x_1
