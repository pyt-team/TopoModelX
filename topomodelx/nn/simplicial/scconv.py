"""Simplicial 2-Complex Convolutional Network Implementation for binary node classification."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.nn.simplicial.scconv_layer import SCConvLayer


class SCConv(torch.nn.Module):
    """Simplicial 2-Complex Convolutional Network Implementation for binary node classification.

    Parameters
    ----------
    node_channels : int
        Dimension of node (0-cells) features.
    edge_channels : int
        Dimension of edge (1-cells) features.
    face_channels : int
        Dimension of face (2-cells) features.
    n_layers : int
        Number of message passing layers.
    n_classes : int
        Number of classes.
    update_func : str
        Activation function used in aggregation layers.

    """

    def __init__(
        self, node_channels, edge_channels, face_channels, n_classes, n_layers=2
    ):
        super().__init__()
        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.face_channels = face_channels
        self.n_classes = n_classes

        layers = []
        for _ in range(n_layers):
            layers.append(
                SCConvLayer(
                    node_channels=node_channels,
                    edge_channels=edge_channels,
                    face_channels=face_channels,
                )
            )

        self.layers = torch.nn.ModuleList(layers)
        self.linear_x0 = torch.nn.Linear(node_channels, self.n_classes)
        self.linear_x1 = torch.nn.Linear(edge_channels, self.n_classes)
        self.linear_x2 = torch.nn.Linear(face_channels, self.n_classes)
        self.aggr = Aggregation(
            aggr_func="mean",
            update_func="sigmoid",
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        incidence_1,
        incidence_1_norm,
        incidence_2,
        incidence_2_norm,
        adjacency_up_0_norm,
        adjacency_up_1_norm,
        adjacency_down_1_norm,
        adjacency_down_2_norm,
    ):
        """Forward computation.

        Parameters
        ----------
        x_0: torch.Tensor, shape = (n_nodes, node_channels)
            Input features on the nodes of the simplicial complex.
        x_1: torch.Tensor, shape = (n_edges, edge_channels)
            Input features on the edges of the simplicial complex.
        x_2: torch.Tensor, shape = (n_faces, face_channels)
            Input features on the faces of the simplicial complex.
        incidence_1: torch.Tensor, shape = (n_faces, channels)
            Incidence matrix of rank 1 :math:`B_1`.
        incidence_1_norm: torch.Tensor
            Normalized incidence matrix of rank 1 :math:`B^{~}_1`.
        incidence_2: torch.Tensor
            Incidence matrix of rank 2 :math:`B_2`.
        incidence_2_norm: torch.Tensor
            Normalized incidence matrix of rank 2 :math:`B^{~}_2`.
        adjacency_up_0_norm: torch.Tensor
            Normalized upper adjacency matrix of rank 0.
        adjacency_up_1_norm: torch.Tensor
            Normalized upper adjacency matrix of rank 1.
        adjacency_down_1_norm: torch.Tensor
            Normalized down adjacency matrix of rank 1.
        adjacency_down_2_norm: torch.Tensor
            Normalized down adjacency matrix of rank 2.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.

        """
        for i in range(self.n_layers):
            x_0, x_1, x_2 = self.layers[i](
                x_0,
                x_1,
                x_2,
                incidence_1,
                incidence_1_norm,
                incidence_2,
                incidence_2_norm,
                adjacency_up_0_norm,
                adjacency_up_1_norm,
                adjacency_down_1_norm,
                adjacency_down_2_norm,
            )

        x_0 = self.linear_x0(x_0)
        x_1 = self.linear_x1(x_1)
        x_2 = self.linear_x2(x_2)

        node_mean = torch.nanmean(x_0, dim=0)
        node_mean[torch.isnan(node_mean)] = 0

        edge_mean = torch.nanmean(x_1, dim=0)
        edge_mean[torch.isnan(edge_mean)] = 0

        face_mean = torch.nanmean(x_2, dim=0)
        face_mean[torch.isnan(face_mean)] = 0

        return (
            torch.flatten(node_mean)
            + torch.flatten(edge_mean)
            + torch.flatten(face_mean)
        )
