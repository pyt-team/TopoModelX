"""High Skip Network Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


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
    initialization : string
        Initialization method.

    """

    def __init__(
        self,
        channels,
    ):
        super().__init__()
        self.channels = channels

        self.conv_level1_0_to_0 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func="sigmoid",
        )
        self.conv_level1_0_to_1 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func="sigmoid",
        )

        self.conv_level2_0_to_0 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func=None,
        )
        self.conv_level2_1_to_0 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func=None,
        )

        self.aggr_on_nodes = Aggregation(aggr_func="sum", update_func="sigmoid")

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_0_to_0.reset_parameters()
        self.conv_level1_0_to_1.reset_parameters()
        self.conv_level2_0_to_0.reset_parameters()
        self.conv_level2_1_to_0.reset_parameters()

    def forward(self, x_0, incidence_1, adjacency_0):
        r"""Forward computation of one layer.

        Parameters
        ----------
        x: torch.tensor
            shape=[n_nodes, channels]
            Input features on the nodes of the simplicial complex.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).
        adjacency_0 : torch.sparse
            shape=[n_nodes, n_nodes]
            Adjacency matrix mapping nodes to nodes (A_0_up).

        Returns
        -------
        _ : torch.tensor
            shape=[n_nodes, channels]
            Output features on the nodes of the simplicial complex.
        """
        incidence_1_transpose = incidence_1.to_dense().T.to_sparse()

        x_0_level1 = self.conv_level1_0_to_0(x_0, adjacency_0)
        x_1_level1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)

        x_0_level2 = self.conv_level2_0_to_0(x_0_level1, adjacency_0)
        x_1_level2 = self.conv_level2_1_to_0(x_1_level1, incidence_1)

        x_0 = self.aggr_on_nodes([x_0_level2, x_1_level2])
        return x_0
