"""HMPNN class."""


import torch

from topomodelx.nn.hypergraph.hmpnn_layer import HMPNNLayer


class HMPNN(torch.nn.Module):
    """Neural network implementation of HMPNN.

    Parameters
    ----------
    in_features : int
        Dimension of input features
    hidden_features : Tuple[int]
        A tuple of hidden feature dimensions to gradually reduce node/hyperedge representations feature
        dimension from in_features to the last item in the tuple.
    num_classes: int
        Number of classes
    n_layer : 2
        Number of HMPNNLayer layers.
    adjacency_dropout_rate: 0.7
        Adjacency dropout rate.
    regular_dropout_rate: 0.5
        Regular dropout rate applied on features.

    References
    ----------
    .. [H22] Heydari S, Livi L.
        Message passing neural networks for hypergraphs.
        International Conference on Artificial Neural Networks 2022 Sep 6 (pp. 583-592). Cham: Springer Nature Switzerland.
        https://arxiv.org/abs/2203.16995
    """

    def __init__(
        self,
        in_features,
        hidden_features,
        num_classes,
        n_layer=2,
        adjacency_dropout_rate=0.7,
        regular_dropout_rate=0.5,
    ):
        super().__init__()
        hidden_features = (in_features,) + hidden_features
        self.to_hidden_linear = torch.nn.Sequential(
            *[
                torch.nn.Linear(hidden_features[i], hidden_features[i + 1])
                for i in range(len(hidden_features) - 1)
            ]
        )
        self.layers = torch.nn.ModuleList(
            [
                HMPNNLayer(
                    hidden_features[-1],
                    adjacency_dropout=adjacency_dropout_rate,
                    updating_dropout=regular_dropout_rate,
                )
                for _ in range(n_layer)
            ]
        )
        self.to_categories_linear = torch.nn.Linear(hidden_features[-1], num_classes)

    def forward(self, x_0, x_1, incidence_1):
        """Forward computation through layers.

        Parameters
        ----------
        x_0 : torch.Tensor
            Node features with shape [n_nodes, in_features]
        x_1 : torch.Tensor
            Hyperedge features with shape [n_hyperedges, in_features]
        incidence_1: torch.sparse.Tensor
            Incidence matrix (B1) of shape [n_nodes, n_hyperedges]

        Returns
        -------
        y_pred : torch.Tensor
            Predicted logits with shape [n_nodes, num_classes]
        """
        x_0 = self.to_hidden_linear(x_0)
        x_1 = self.to_hidden_linear(x_1)
        for layer in self.layers:
            x_0, x_1 = layer(x_0, x_1, incidence_1)

        return self.to_categories_linear(x_0)
