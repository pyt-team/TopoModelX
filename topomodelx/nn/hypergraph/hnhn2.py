"""HNHN class."""

import torch

from topomodelx.nn.hypergraph.hnhn2_layer import HNHN2Layer


class HNHN2(torch.nn.Module):
    """Neural network implementation of HNHN.

    Parameters
    ----------
    in_features : int
        Dimension of input features
    hidden_features : int
        Dimension of hidden features
    incidence_1: torch.sparse.Tensor
        Incidence matrix of shape [n_nodes, n_hyperedges]
    num_classes: int
        Number of classes
    n_layer : 2
        Number of HNHNLayer layers.
    dropout_rate: 0.3

    References
    ----------
    .. [DSB20] Dong, Sawin, Bengio.
        HNHN: Hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020
        https://grlplus.github.io/papers/40.pdf
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        incidence_1,
        num_classes,
        n_layer=2,
        dropout_rate=0.3,
    ):
        super().__init__()
        self.to_hidden_linear = torch.nn.Linear(in_features, hidden_features)
        self.layers = torch.nn.Sequential(
            *[
                HNHN2Layer(
                    hidden_features,
                    incidence_1,
                    normalization_param_alpha=-1.5,
                    normalization_param_beta=-0.5,
                )
                for _ in range(n_layer)
            ]
        )
        self.dropout = torch.nn.Dropout(dropout_rate)
        self.to_categories_linear = torch.nn.Linear(hidden_features, num_classes)

    def forward(self, x_0):
        """Forward computation through layers.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features with shape [n_nodes, in_features]

        Returns
        -------
        y_pred : torch.Tensor
            Predicted logits with shape [n_nodes, num_classes]
        x_1: torch.Tensor
            Final hidden representation of hyperedges with shape [n_hyperedges, hidden_features]
        """
        x_1 = 0
        x_0 = self.to_hidden_linear(x_0)
        for i, layer in enumerate(self.layers):
            x_0, x_1 = layer(x_0)
            if i != len(self.layers) - 1:
                x_0, x_1 = self.dropout(x_0), self.dropout(x_1)

        return self.to_categories_linear(x_0), x_1
