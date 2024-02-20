"""UniGCNII class."""

import torch

from topomodelx.nn.hypergraph.unigin_layer import UniGINLayer


class UniGIN(torch.nn.Module):
    """Neural network implementation of UniGIN [1]_ for hypergraph classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default = 2
        Amount of message passing layers.
    input_drop : float, default=0.2
        Dropout rate for the input features.
    layer_drop : float, default=0.2
        Dropout rate for the hidden features.


    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        input_drop=0.2,
        layer_drop=0.2,
    ):
        super().__init__()

        self.input_drop = torch.nn.Dropout(input_drop)
        self.layer_drop = torch.nn.Dropout(layer_drop)

        self.initial_linear_layer = torch.nn.Linear(in_channels, hidden_channels)

        self.layers = torch.nn.ModuleList(
            UniGINLayer(
                in_channels=hidden_channels,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_node)
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
        x_0 = self.initial_linear_layer(x_0)
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1)
            x_0 = self.layer_drop(x_0)
            x_0 = torch.nn.functional.relu(x_0)

        return x_0, x_1
