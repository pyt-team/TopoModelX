"""UniGCNII class."""

import math

import torch

from topomodelx.nn.hypergraph.unigcnii_layer import UniGCNIILayer


class UniGCNII(torch.nn.Module):
    """Hypergraph neural network utilizing the UniGCNII layer [1]_ for node-level classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layers: int, default=2
        Number of UniGCNII message passing layers.
    alpha : float, default=0.5
        Parameter of the UniGCNII layer.
    beta : float, default=0.5
        Parameter of the UniGCNII layer.
    input_drop: float, default=0.2
        Dropout rate for the input features.
    layer_drop: float, default=0.2
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
        alpha=0.5,
        beta=0.5,
        input_drop=0.2,
        layer_drop=0.2,
    ):
        super().__init__()
        layers = []

        self.input_drop = torch.nn.Dropout(input_drop)
        self.layer_drop = torch.nn.Dropout(layer_drop)
        # Define initial linear layer
        self.linear_init = torch.nn.Linear(in_channels, hidden_channels)

        # Define convolutional layers
        for i in range(n_layers):
            beta = math.log(alpha / (i + 1) + 1)
            layers.append(
                UniGCNIILayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    alpha=alpha,
                    beta=beta,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, incidence_1):
        """Forward pass through the model.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels)
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = (num_nodes, num_edges)
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        x_0 = self.input_drop(x_0)
        x_0 = self.linear_init(x_0)
        x_0 = torch.nn.functional.relu(x_0)
        x_0_skip = x_0

        for layer in self.layers:
            x_0, x_1 = layer(x_0, incidence_1, x_0_skip)
            x_0 = self.layer_drop(x_0)
            x_0 = torch.nn.functional.relu(x_0)

        return (x_0, x_1)
