"""UniGCN class."""

import torch

from topomodelx.nn.hypergraph.unigcn_layer import UniGCNLayer


class UniGCN(torch.nn.Module):
    """Neural network implementation of UniGCN [1]_ for hypergraph classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layer : int, default = 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(self, 
        in_channels,
        hidden_channels,
        n_layers=2,
    ):
        super().__init__()
        layers = []
        layers.append(
                UniGCNLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                )
            )
        for _ in range(n_layers - 1):
            layers.append(
                UniGCNLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                )
            )
        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_edges, channels_edge)
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
            x_0, x_1 = layer(x_0, incidence_1)
        
        return (x_0, x_1)