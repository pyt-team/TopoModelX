"""HNHN class."""

import torch

from topomodelx.nn.hypergraph.hnhn_layer import HNHNLayer


class HNHN(torch.nn.Module):
    """Hypergraph Networks with Hyperedge Neurons [1]_. Implementation for multiclass node classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    incidence_1 : torch.sparse, shape = (n_nodes, n_edges)
        Incidence matrix mapping edges to nodes (B_1).
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
        self, 
        in_channels, 
        hidden_channels, 
        incidence_1,  
        n_layers=2
    ):
        super().__init__()

        layers = []
        layers.append(
                HNHNLayer(
                    in_channels=in_channels,
                    hidden_channels=hidden_channels,
                    incidence_1=incidence_1,
                )
            )
        for _ in range(n_layers - 1):
            layers.append(
                HNHNLayer(
                    in_channels=hidden_channels,
                    hidden_channels=hidden_channels,
                    incidence_1=incidence_1,
                )
            )
        self.layers = torch.nn.ModuleList(layers)


    def forward(self, x_0, x_1):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels_node)
            Hypernode features.

        x_1 : torch.Tensor, shape = (n_nodes, channels_edge)
            Hyperedge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        
        for layer in self.layers:
            x_0, x_1 = layer(x_0, x_1)
        
        return x_0, x_1



# class HNHNNetwork(torch.nn.Module):
#     """Hypergraph Networks with Hyperedge Neurons. Implementation for multiclass node classification.

#     Parameters
#     ----------
#     channels_node : int
#         Dimension of node features.
#     channels_edge : int
#         Dimension of edge features.
#     incidence_1 : torch.sparse
#         Incidence matrix mapping edges to nodes (B_1).
#         shape=[n_nodes, n_edges]
#     n_classes: int
#         Number of classes
#     n_layers : int
#         Number of HNHN message passing layers.
#     """

#     def __init__(
#         self, channels_node, channels_edge, incidence_1, n_classes, n_layers=2
#     ):
#         super().__init__()
#         self.layers = torch.nn.ModuleList(
#             [
#                 HNHNLayer(
#                     channels_node=channels_node,
#                     channels_edge=channels_edge,
#                     incidence_1=incidence_1,
#                 )
#                 for _ in range(n_layers)
#             ]
#         )
#         self.linear = torch.nn.Linear(channels_node, n_classes)

#     def forward(self, x_0, x_1):
#         """Forward computation.

#         Parameters
#         ----------
#         x_0 : torch.Tensor
#             shape = [n_nodes, channels_node]
#             Hypernode features.

#         x_1 : torch.Tensor
#             shape = [n_nodes, channels_edge]
#             Hyperedge features.

#         incidence_1 : torch.Tensor
#             shape = [n_nodes, n_edges]
#             Boundary matrix of rank 1.

#         Returns
#         -------
#         x_0 : torch.Tensor
#             Output node features.
#         x_1 : torch.Tensor
#             Output hyperedge features.
#         """
        
#         for layer in self.layers:
#             x_0, x_1 = layer(x_0, x_1)
        
#         return x_0, x_1
