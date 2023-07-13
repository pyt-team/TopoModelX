"""Higher Order Attention Network Layer."""
import torch
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class HOANMCLayer(torch.nn.Module):
    """Layer of a Higher Order Attention Network.

    Implementation of the HOAN layer proposed for mesh classification in [HZPMG22]_.

    Notes
    -----
    This is the architecture proposed for mesh classification on combinatorial complices.

    References
    ----------
    .. [HZPMG22] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz.
        Higher-Order Attention Networks.
        arXiv preprint arXiv:2206.00606, 2022a.

    Parameters
    ----------
    channels : list[int]
        Dimension of features on each combinatorial cell.
    """

    def __init__(
        self,
        channels,
    ):
        super().__init__()
        self.channels = channels

        self.conv_level1_0_to_0 = Conv(
            in_channels=channels[0],
            out_channels=channels[0],
            update_func=None,
            att=True,
        )
        self.conv_level1_0_to_1 = Conv(
            in_channels=channels[0],
            out_channels=channels[1],
            update_func=None,
            att=True,
        )
        self.conv_level1_1_to_0 = Conv(
            in_channels=channels[1],
            out_channels=channels[0],
            update_func=None,
            att=True,
        )
        self.conv_level1_1_to_2 = Conv(
            in_channels=channels[1],
            out_channels=channels[2],
            update_func=None,
            att=True,
        )
        self.conv_level1_2_to_1 = Conv(
            in_channels=channels[2],
            out_channels=channels[1],
            update_func=None,
            att=True,
        )

        self.aggr = Aggregation(
            aggr_func="sum",
            update_func=None,
        )

        self.conv_level2_0_to_0 = Conv(
            in_channels=channels[0],
            out_channels=channels[0],
            update_func=None,
            att=True,
        )
        self.conv_level2_0_to_1 = Conv(
            in_channels=channels[0],
            out_channels=channels[1],
            update_func=None,
            att=True,
        )
        self.conv_level2_1_to_1 = Conv(
            in_channels=channels[1],
            out_channels=channels[1],
            update_func=None,
            att=True,
        )
        self.conv_level2_1_to_2 = Conv(
            in_channels=channels[1],
            out_channels=channels[2],
            update_func=None,
            att=True,
        )
        self.conv_level2_2_to_2 = Conv(
            in_channels=channels[2],
            out_channels=channels[2],
            update_func=None,
            att=True,
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_0_to_0.reset_parameters()
        self.conv_level1_0_to_1.reset_parameters()
        self.conv_level1_1_to_0.reset_parameters()
        self.conv_level1_1_to_2.reset_parameters()
        self.conv_level1_2_to_1.reset_parameters()

        self.conv_level2_0_to_0.reset_parameters()
        self.conv_level2_0_to_1.reset_parameters()
        self.conv_level2_1_to_1.reset_parameters()
        self.conv_level2_1_to_2.reset_parameters()
        self.conv_level2_2_to_2.reset_parameters()

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        up_adjacency_0,
        incidence_1,
        up_adjacency_1,
        incidence_2,
        down_adjacency_2,
    ):
        r"""Forward pass.

        The forward pass was initially proposed in [HZPMG22]_.
        Its equations are graphically illustrated in [PSHM23]_.

        References
        ----------
        .. [HZPMG22] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz.
            Higher-Order Attention Networks.
            arXiv preprint arXiv:2206.00606, 2022a.
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0: torch.Tensor, shape=[n_nodes, channels[0]]
            Input features on the nodes of the combinatorial complex.
        x_1: torch.Tensor, shape=[n_1cells, channels[1]]
            Input features on the one-cells of the combinatorial complex.
        x_2: torch.Tensor, shape=[n_2cells, channels[2]]
            Input features on the two-cells of the combinatorial complex.
        up_adjacency_0: torch.sparse, shape=[n_nodes, n_1cells]
            Adjacency matrix: :math: `A_0^{\uparrow}` mapping nodes to nodes via one-cells.
        incidence_1: torch.sparse, shape=[n_1cells, n_nodes]
            Incidence matrix: :math:`B_1` mapping one-cells to nodes.
        up_adjacency_1: torch.sparse, shape=[n_1cells, n_1cells]
            Adjacency matrix: :math: `A_1^{\uparrow}` mapping one-cells to one-cells via two-cells.
        incidence_2: torch.sparse, shape=[n_2cells, n_1cells]
            Incidence matrix: :math:`B_2` mapping two-cells to one-cells.
        down_adjacency_2: torch.sparse, shape=[n_2cells, n_2cells]
            Adjacency matrix: :math: `A_2^{\downarrow}` mapping two-cells to two-cells via one-cells.

        Returns
        -------
        x_0: torch.Tensor, shape=[n_nodes, channels[0]]
            Output features on the nodes of the combinatorial complex.
        x_1: torch.Tensor, shape=[n_1cells, channels[1]]
            Input features on the one-cells of the combinatorial complex.
        x_2: torch.Tensor, shape=[n_2cells, channels[2]]
            Output features on the two-cells of the combinatorial complex.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)
        incidence_2_transpose = incidence_2.transpose(1, 0)

        x_0_level1_0 = self.conv_level1_0_to_0(x_0, up_adjacency_0, x_target=x_0)
        x_1_level1_0 = self.conv_level1_0_to_1(x_0, incidence_1_transpose, x_target=x_1)
        x_0_level1_1 = self.conv_level1_1_to_0(x_1, incidence_1, x_target=x_0)
        x_1_level1_2 = self.conv_level1_2_to_1(x_2, incidence_2, x_target=x_1)
        x_2_level1 = self.conv_level1_1_to_2(x_1, incidence_2_transpose, x_target=x_2)

        x_0_level1 = self.aggr([x_0_level1_0, x_0_level1_1])
        x_1_level1 = self.aggr([x_1_level1_0, x_1_level1_2])

        x_0_level2_0 = self.conv_level2_0_to_0(
            x_0_level1, up_adjacency_0, x_target=x_0_level1
        )
        x_1_level2_0 = self.conv_level2_0_to_1(
            x_0_level1, incidence_1_transpose, x_target=x_1_level1
        )
        x_1_level2_1 = self.conv_level2_1_to_1(
            x_1_level1, up_adjacency_1, x_target=x_1_level1
        )
        x_2_level2_1 = self.conv_level2_1_to_2(
            x_1_level1, incidence_2_transpose, x_target=x_2_level1
        )
        x_2_level2_2 = self.conv_level2_2_to_2(
            x_2_level1, down_adjacency_2, x_target=x_2_level1
        )

        x_0 = x_0_level2_0
        x_1 = self.aggr([x_1_level2_0, x_1_level2_1])
        x_2 = self.aggr([x_2_level2_1, x_2_level2_2])

        return x_0, x_1, x_2
