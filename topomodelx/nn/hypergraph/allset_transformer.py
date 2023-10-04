"""Allset transformer class."""

import torch

from topomodelx.nn.hypergraph.allset_transformer_layer import AllSetTransformerLayer


class AllSetTransformer(torch.nn.Module):
    """AllSet Neural Network Module.

    A module that combines multiple AllSet layers [1]_ to form a neural network.

    Parameters
    ----------
    in_dim : int
        Dimension of the input features.
    hid_dim : int
        Dimension of the hidden features.
    out_dim : int
        Dimension of the output features.
    dropout : float
        Dropout probability.
    n_layers : int, default: 2
        Number of AllSet layers in the network.
    input_dropout : float, default: 0.2
        Dropout probability for the layer input.
    mlp_num_layers : int, default: 2
        Number of layers in the MLP.
    mlp_norm : bool, optional. default: False
        Whether to apply input normalization in the MLP.

    References
    ----------
    .. [1] Chien, Pan, Peng and Milenkovic.
        You are AllSet: a multiset function framework for hypergraph neural networks.
        ICLR 2022.
        https://arxiv.org/abs/2106.13264
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers=1,
        heads=4,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_dropout=0.0,
    ):
        super().__init__()
        layers = [
            AllSetTransformerLayer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                dropout=dropout,
                heads=heads,
                mlp_num_layers=mlp_num_layers,
                mlp_dropout=mlp_dropout,
            )
        ]

        for _ in range(n_layers - 1):
            layers.append(
                AllSetTransformerLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    heads=heads,
                    mlp_num_layers=mlp_num_layers,
                    mlp_dropout=mlp_dropout,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_0, incidence_1):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        edge_index : torch.Tensor
            Edge list (of size (2, |E|)).

        Returns
        -------
        torch.Tensor
            Output prediction.
        """
        for layer in self.layers:
            x_0 = layer(x_0, incidence_1)
        pooled_x = torch.max(x_0, dim=0)[0]
        return torch.sigmoid(self.linear(pooled_x))[0]
