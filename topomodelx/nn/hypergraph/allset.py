"""Allset class."""

import torch

from topomodelx.nn.hypergraph.allset_layer import AllSetLayer


class AllSet(torch.nn.Module):
    """AllSet Neural Network Module.

    A module that combines multiple AllSet layers [1]_ to form a neural network.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers : int, default: 2
        Number of AllSet layers in the network.
    layer_dropout: float, default: 0.2
        Dropout probability for the AllSet layer.
    mlp_num_layers : int, default: 2
        Number of layers in the MLP.
    mlp_dropout : float, default: 0.0
        Dropout probability for the MLP.
    mlp_activation : torch.nn.Module, default: None
        Activation function in the MLP.
    mlp_norm : bool, default: False
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
        n_layers=2,
        layer_dropout=0.2,
        mlp_num_layers=2,
        mlp_activation=None,
        mlp_dropout=0.0,
        mlp_norm=None,
    ):
        super().__init__()
        layers = [
            AllSetLayer(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                dropout=layer_dropout,
                mlp_num_layers=mlp_num_layers,
                mlp_activation=mlp_activation,
                mlp_dropout=mlp_dropout,
                mlp_norm=mlp_norm,
            )
        ]

        for _ in range(n_layers - 1):
            layers.append(
                AllSetLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    dropout=layer_dropout,
                    mlp_num_layers=mlp_num_layers,
                    mlp_activation=mlp_activation,
                    mlp_dropout=mlp_dropout,
                    mlp_norm=mlp_norm,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, incidence_1):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features.
        incidence_1 : torch.Tensor
            Edge list (of size (2, |E|)).

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1)

        return x_0, x_1
