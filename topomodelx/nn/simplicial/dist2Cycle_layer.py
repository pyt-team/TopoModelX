"""High Skip Network Layer."""
import torch

import torch.nn as nn
import torch.nn.functional as F

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class Dist2CycleLayer(torch.nn.Module):
    """Layer of a High Skip Network (Dist2Cycle)."""

    def __init__(
        self,
        channels,  # in_feats
        # in_feats=8,
        # out_feats=1,
    ):
        super().__init__()
        self.channels = channels
        # feature learning
        self.fc_neigh = nn.Linear(channels, 1, bias=True)
        self.aggr_on_edges = Aggregation(
            aggr_func="sum", update_func="relu"
        )  # need to support for other update functions like leaky relu which is main for dist2Cycle

        # self.conv_level1_0_to_0 = Conv(
        #     in_channels=channels,
        #     out_channels=channels,
        #     update_func="sigmoid",
        # )
        # self.conv_level1_0_to_1 = Conv(
        #     in_channels=channels,
        #     out_channels=channels,
        #     update_func="sigmoid",
        # )

        # self.conv_level2_0_to_0 = Conv(
        #     in_channels=channels,
        #     out_channels=channels,
        #     update_func=None,
        # )
        # self.conv_level2_1_to_0 = Conv(
        #     in_channels=channels,
        #     out_channels=channels,
        #     update_func=None,
        # )

    def reset_parameters(self):
        r"""Reset learnable parameters."""

        fc_nonlin = "relu"
        fc_alpha = 0.0  # self.fc_activation.negative_slope
        fc_gain = nn.init.calculate_gain(fc_nonlin)
        self.fc_neigh.reset_parameters()

        nn.init.kaiming_uniform_(
            self.fc_neigh.weight, a=fc_alpha, nonlinearity=fc_nonlin
        )

    # self.conv_level1_0_to_0.reset_parameters()
    # self.conv_level1_0_to_1.reset_parameters()
    # self.conv_level2_0_to_0.reset_parameters()
    # self.conv_level2_1_to_0.reset_parameters()

    def forward(self, x_e, Linv, adjacency):
        r"""Forward pass.


        Parameters
        ----------
        x: torch.Tensor, shape=[n_nodes, channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse, shape=[n_nodes, n_edges]
            Incidence matrix :math:`B_1` mapping edges to nodes.
        adjacency_0 : torch.sparse, shape=[n_nodes, n_nodes]
            Adjacency matrix :math:`A_0^{\uparrow}` mapping nodes to nodes via edges.

        Returns
        -------
        _ : torch.Tensor, shape=[n_nodes, channels]
            Output features on the nodes of the simplicial complex.
        """
        # incidence_1_transpose = incidence_1.transpose(1, 0)

        # x_0_level1 = self.conv_level1_0_to_0(x_0, adjacency_0)
        # x_1_level1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)

        # x_0_level2 = self.conv_level2_0_to_0(x_0_level1, adjacency_0)
        # x_1_level2 = self.conv_level2_1_to_0(x_1_level1, incidence_1)
        # print("adjacency")
        # print(adjacency.to_dense().shape)
        # print("Linv")
        # print(Linv.to_dense().shape)

        # x_e = (adjacency.to_dense()*Linv.to_dense()).to_sparse()
        x_e = adjacency * Linv
        # print("x_e")
        # print(x_e.to_dense().shape)
        # print("Test")
        # u, s, v = torch.svd(x_t.to_dense())
        # print(u.shape)
        # print(s.shape)
        # print(v.shape)
        # x_t2 = u[:, :2].double().to_sparse()
        # x_t2 = torch.zeros_like(x_e).to_sparse()
        # x_e = self.aggr_on_edges([x_e, x_e])
        # x_t2=torch
        x_e = self.aggr_on_edges([x_e])
        rst = self.fc_neigh(x_e)
        # print("rst")
        # print(rst.shape)
        return rst
