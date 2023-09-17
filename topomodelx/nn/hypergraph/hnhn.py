"""HNHN class."""

import torch

from topomodelx.nn.hypergraph.hnhn_layer import HNHNLayer


class HNHN(torch.nn.Module):
    """Hypergraph Networks with Hyperedge Neurons [1]_. Implementation for multiclass node classification.

    Parameters
    ----------
    channels_node : int
        Dimension of node features.
    channels_edge : int
        Dimension of edge features.
    incidence_1 : torch.sparse
        Incidence matrix mapping edges to nodes (B_1).
        shape=[n_nodes, n_edges]
    n_classes: int
        Number of classes
    n_layers : int
        Number of HNHN message passing layers.

    References
    ----------
    .. [1] Dong, Sawin, Bengio.
        HNHN: hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020.
        https://grlplus.github.io/papers/40.pdf
    """

    def __init__(
        self, channels_node, channels_edge, incidence_1, n_classes, n_layers=2
    ):
        super().__init__()
        self.layers = torch.nn.ModuleList(
            [
                HNHNLayer(
                    channels_node=channels_node,
                    channels_edge=channels_edge,
                    incidence_1=incidence_1,
                )
                for _ in range(n_layers)
            ]
        )
        self.linear = torch.nn.Linear(channels_node, n_classes)

    def forward(self, x_0, x_1):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor
            shape = [n_nodes, channels_node]
            Hypernode features.

        x_1 : torch.Tensor
            shape = [n_nodes, channels_edge]
            Hyperedge features.

        incidence_1 : tensor
            shape = [n_nodes, n_edges]
            Boundary matrix of rank 1.

        Returns
        -------
        logits : torch.Tensor
            The predicted node logits
            shape = [n_nodes, n_classes]
        classes : torch.Tensor
            The predicted node class
            shape = [n_nodes]
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0, x_1)
        logits = self.linear(x_0)
        classes = torch.softmax(logits, -1).argmax(-1)
        return logits, classes
