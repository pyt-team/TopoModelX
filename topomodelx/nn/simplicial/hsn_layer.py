import torch

from topomodelx.base.conv import MessagePassingConv
from topomodelx.base.merge import _Merge


class HSNLayer(torch.nn.Module):
    """High Skip Network Layer.
    Implementation of the HSN layer from the paper by Hajij et. al:
    High Skip Networks: A Higher Order Generalization of Skip Connections
    https://openreview.net/pdf?id=Sc8glB-k6e9
    Note: this is the architecture proposed for node classification on simplicial complices.

    Parameters
    ----------
    channels : int
        Dimension of features on each simplicial cell.
    incidence_matrix_1 : torch.sparse
        Incidence matrix mapping edges to nodes (B_1).
    adjacency_matrix_0 : torch.sparse
        Adjacency matrix mapping nodes to nodes (A_0_up).
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        channels,
        incidence_matrix_1,
        adjacency_matrix_0,
    ):
        super().__init__()
        self.channels = channels
        self.incidence_matrix_1 = incidence_matrix_1
        incidence_matrix_1_transpose = incidence_matrix_1.to_dense().T.to_sparse()
        self.adjacency_matrix_0 = adjacency_matrix_0

        self.message_passing_level1_0_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=adjacency_matrix_0,
            update_on_message="sigmoid",
        )
        self.message_passing_level1_0_to_1 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=incidence_matrix_1_transpose,
            update_on_message="sigmoid",
        )

        self.message_passing_level2_0_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=adjacency_matrix_0,
            update_on_message=None,
        )
        self.message_passing_level2_1_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=incidence_matrix_1,
            update_on_message=None,
        )

        self.merge_on_nodes = _Merge(inter_aggr="sum", update_on_merge="sigmoid")

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.message_passing_level1_0_to_0.reset_parameters()
        self.message_passing_level1_0_to_1.reset_parameters()
        self.message_passing_level2_0_to_0.reset_parameters()
        self.message_passing_level2_1_to_0.reset_parameters()

    def forward(self, x):
        r"""Forward computation of one layer.

        Parameters
        ----------
        x: torch.tensor
            shape=[n_nodes, channels]
            Input features on the nodes of the simplicial complex.
        """
        x_nodes_level1 = self.message_passing_level1_0_to_0(x)
        x_edges_level1 = self.message_passing_level1_0_to_1(x)

        x_nodes_level2 = self.message_passing_level2_0_to_0(x_nodes_level1)
        x_edges_level2 = self.message_passing_level2_1_to_0(x_edges_level1)

        x = self.merge_on_nodes([x_nodes_level2, x_edges_level2])
        return x
