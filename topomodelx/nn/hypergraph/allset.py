"""Allset class."""

import torch

from topomodelx.nn.hypergraph.allset_layer import AllSetLayer


class AllSet(torch.nn.Module):
    """AllSet Neural Network Module.

    A module that combines multiple AllSet layers to form a neural network.

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
    n_layers : int, optional
        Number of AllSet layers in the network. Defaults to 2.
    input_dropout : float, optional
        Dropout probability for the layer input. Defaults to 0.2.
    mlp_num_layers : int, optional
        Number of layers in the MLP. Defaults to 2.
    mlp_norm : bool, optional
        Whether to apply input normalization in the MLP. Defaults to False.

    References
    ----------
    .. [E21] Eli Chien, Chao Pan, Jianhao Peng, Olgica Milenkovic.
        You are AllSet: A Multiset Function Framework for Hypergraph Neural Networks. (2021)
        https://arxiv.org/abs/2106.13264
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_layers=2,
        dropout=0.2,
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
                dropout=dropout,
                mlp_num_layers=mlp_num_layers,
                mlp_activation=mlp_activation,
                mlp_dropout=mlp_dropout,
                mlp_norm=mlp_norm,
            )
        ]

        for _ in range(n_layers - 1):
            layers.append(
                AllSetLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    dropout=dropout,
                    mlp_num_layers=mlp_num_layers,
                    mlp_activation=mlp_activation,
                    mlp_dropout=mlp_dropout,
                    mlp_norm=mlp_norm,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x_0, incidence_1):
        """Forward computation.

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
        # cidx = edge_index[1].min()
        # edge_index[1] -= cidx
        # reversed_edge_index = torch.stack(
        #     [edge_index[1], edge_index[0]], dim=0)

        for layer in self.layers:
            x_0 = layer(x_0, incidence_1)
        pooled_x = torch.max(x_0, dim=0)[0]
        return torch.sigmoid(self.linear(pooled_x))[0]
