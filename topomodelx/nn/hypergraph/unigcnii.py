"""UniGCNII class."""

import torch

from topomodelx.nn.hypergraph.unigcnii_layer import UniGCNIILayer


class UniGCNII(torch.nn.Module):
    """Hypergraph neural network utilizing the UniGCNII layer [1]_ for node-level classification.

    Parameters
    ----------
    num_classes: int, default=2
        Number of classes used for node classification.
    in_features: int, default=1
        Number of input features on the nodes.
    n_layers: int, default=2
        Number of UniGCNII message passing layers.
    alpha: float, default=0.5
        Parameter of the UniGCNII layer.
    beta: float, default=0.5
        Parameter of the UniGCNII layer.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(self, num_classes=2, in_features=1, num_layers=2, alpha=0.5, beta=0.5):
        super().__init__()
        layers = []
        self.num_features = in_features
        self.num_classes = num_classes

        for _ in range(num_layers):
            layers.append(
                UniGCNIILayer(in_channels=in_features, alpha=alpha, beta=beta)
            )

        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(self.num_features, self.num_classes)

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
        y_hat : torch.Tensor, shape = (num_nodes, num_classes)
            Contains the logits for classification for every node.
        """
        # Copy the original features to use as skip connections
        x_0_skip = x_0.clone()

        for layer in self.layers:
            x_0 = layer(x_0, incidence_1, x_0_skip)

        # linear layer for node classification output
        # softmax ommited for use of cross-entropy loss
        return self.linear(x_0)
