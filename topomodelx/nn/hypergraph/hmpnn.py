"""HMPNN class."""


import torch

from topomodelx.nn.hypergraph.hmpnn_layer import HMPNNLayer


class HMPNN(torch.nn.Module):
    """Neural network implementation of HMPNN [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of input features
    hidden_channels : Tuple[int]
        A tuple of hidden feature dimensions to gradually reduce node/hyperedge representations feature
        dimension from in_features to the last item in the tuple.
    num_classes: int
        Number of classes
    n_layers : int, default = 2
        Number of HMPNNLayer layers.
    adjacency_dropout_rate: int, default = 0.7
        Adjacency dropout rate.
    regular_dropout_rate: int, default = 0.5
        Regular dropout rate applied on features.

    References
    ----------
    .. [1] Heydari S, Livi L.
        Message passing neural networks for hypergraphs.
        ICANN 2022.
        https://arxiv.org/abs/2203.16995
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        n_layers=2,
        adjacency_dropout_rate=0.7,
        regular_dropout_rate=0.5,
    ):
        super().__init__()

        self.linear_node = torch.nn.Linear(in_channels, hidden_channels)
        self.linear_edge = torch.nn.Linear(in_channels, hidden_channels)
        # self.to_hidden_linear = torch.nn.Sequential(
        #     *[
        #         torch.nn.Linear(hidden_features[i], hidden_features[i + 1])
        #         for i in range(len(hidden_features) - 1)
        #     ]
        # )

        self.layers = torch.nn.ModuleList(
            [
                HMPNNLayer(
                    hidden_channels,
                    adjacency_dropout=adjacency_dropout_rate,
                    updating_dropout=regular_dropout_rate,
                )
                for _ in range(n_layers)
            ]
        )
        #self.to_categories_linear = torch.nn.Linear(hidden_features[-1], num_classes)

    def forward(self, x_0, x_1, incidence_1):
        """Forward computation through layers.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_features)
            Node features.
        x_1 : torch.Tensor, shape = (n_hyperedges, in_features)
            Hyperedge features.
        incidence_1: torch.sparse.Tensor, shape = (n_nodes, n_hyperedges)
            Incidence matrix (B1).

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        x_0 = self.linear_node(x_0)
        x_1 = self.linear_edge(x_1)

        for layer in self.layers:
            x_0, x_1 = layer(x_0, x_1, incidence_1)

        return (x_0, x_1)
