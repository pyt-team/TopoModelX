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
    out_channels : int
        Dimension of the output features.
    dropout : float
        Dropout probability.
    n_layers : int, default: 2
        Number of AllSet layers in the network.
    input_dropout : float, default: 0.2
        Dropout probability for the layer input.
    mlp_num_layers : int, default: 2
        Number of layers in the MLP.
    mlp_norm : bool, default: False
        Whether to apply input normalization in the MLP.
    task_level: str, default="graph"
        Level of the task. Either "graph" or "node".
        If "graph", the output is pooled over all nodes in the hypergraph.

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
        n_layers=2,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_activation=None,
        mlp_dropout=0.0,
        mlp_norm=None,
        task_level="graph",
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
                    in_channels=hidden_channels,
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
        self.out_pool = True if task_level == "graph" else False

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
        
        for layer in self.layers:
            x_0 = layer(x_0, incidence_1)
        
        # Pool over all nodes in the hypergraph 
        if self.out_pool is True:
            x = torch.max(x_0, dim=0)[0]
        else:
            x = x_0

        return self.linear(x)
