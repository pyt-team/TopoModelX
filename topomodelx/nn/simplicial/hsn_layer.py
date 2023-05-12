import torch
import torch.nn.functional as F

from topomodelx.base.level import Level
from topomodelx.base.merge import _Merge
from topomodelx.nn.conv import MessagePassingConv


class HSNLayer(torch.nn.Module):
    """Template Layer.

    Parameters
    ----------
    channels : int
        Dimension of input features.
    channels : int
        Dimension of output features.
    intra_aggr : string
        Aggregation method.
        (Inter-neighborhood).
    """

    def __init__(
        self,
        channels,
        incidence_matrix_1,
        adjacency_matrix_0,
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.channels = channels
        self.incidence_matrix_1 = incidence_matrix_1
        incidence_matrix_1_transpose = incidence_matrix_1.to_dense().T.to_sparse()
        self.adjacency_matrix_0 = adjacency_matrix_0
        self.initialization = initialization

        self.message_passing_level1_0_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=adjacency_matrix_0,
            update="sigmoid",
        )
        self.message_passing_level1_0_to_1 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=incidence_matrix_1_transpose,
            update="sigmoid",
        )

        self.message_passing_level2_0_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=adjacency_matrix_0,
            update=None,
        )
        self.message_passing_level2_1_to_0 = MessagePassingConv(
            in_channels=channels,
            out_channels=channels,
            neighborhood=incidence_matrix_1,
            update=None,
        )

        self.merge_on_nodes = _Merge(inter_aggr="sum", update_on_merge="sigmoid")

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.message_passing_level1_0_to_0.reset_parameters()
        self.message_passing_level1_0_to_1.reset_parameters()
        self.message_passing_level2_0_to_0.reset_parameters()
        self.message_passing_level2_1_to_0.reset_parameters()

    def forward(self, x):
        r"""Forward computation.

        Parameters
        ----------
        x: torch.tensor, shape=[n_nodes, channels]
            Input features on the nodes of the simplicial complex.
        """
        x_nodes_level1 = self.message_passing_level1_0_to_0(x)
        x_edges_level1 = self.message_passing_level1_0_to_1(x)

        x_nodes_level2 = self.message_passing_level2_0_to_0(x_nodes_level1)
        x_edges_level2 = self.message_passing_level2_1_to_0(x_edges_level1)

        x_nodes = self.merge_on_nodes([x_nodes_level2, x_edges_level2])
        return x_nodes
