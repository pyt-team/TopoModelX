"""Allset transformer class."""

import torch

from topomodelx.nn.hypergraph.allset_transformer_layer import AllSetTransformerLayer


class AllSetTransformer(torch.nn.Module):
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
    .. [ECCP22] Chien, E., Pan, C., Peng, J., & Milenkovic, O. You are AllSet: A Multiset
      Function Framework for Hypergraph Neural Networks. In International Conference on
      Learning Representations, 2022 (https://arxiv.org/pdf/2106.13264.pdf)
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
