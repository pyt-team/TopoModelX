import torch
import toponetx as tnx
from topomodelx.base.aggregation import Aggregation
from typing import List

from topomodelx.base.hbs import HBS
from topomodelx.base.hbns import HBNS


class HMCLayer(torch.nn.Module):
    """Layer of a Combinatorial Complex Attention Neural Network for Mesh Classification.

    Implementation of the Combinatorial Complex Attention Neural Network layer for mesh classification
    introduced in [HAJIJ23]_, Figure 35(b).

    This layer works with combinatorial complexes of dimension two. It is composed of two message passing steps,
    meaning that all cochains in cells are updated twice. Message passing is performed using combinatorial complex
    attention push-forward operations. See Definitions 32 and 33 in [HAJIJ23]_ for more details.
    message passing. The steps are:
    1. 0-dimensional cells (vertices) receive messages from 0-dimensional cells (vertices) and from 1-dimensional cells
    (edges). In the first case, adjacency matrices are used. In the second case, the incidence matrix from dimension 1
    to dimension 0 is used. 1-dimensional cells (edges) receive messages from 1-dimensional cells (edges) and from
    2-dimensional cells (faces). In both cases, incidence matrices are used. 2-dimensional cells (faces) receive
    messages only from 1-dimensional cells (edges). In this case, the incidence matrix from dimension 2 to dimension 1
    is used.
    2. 0-dimensional cells (vertices) receive messages from 0-dimensional cells (vertices) using their adjacency matrix.
    1-dimensional cells (edges) receive messages from 0-dimensional cells (vertices) and from
    1-dimensional cells (edges) using incidence and adjacency matrices, respectively. 2-dimensional cells (faces)
    receive messages from 1-dimensional cells (edges) and from 2-dimensional cells (faces) using
    incidence and coadjacency matrices, respectively.

    References
    ----------
    .. [HAJIJ23] Mustafa Hajij et al. Topological Deep Learning: Going Beyond Graph Data.
        arXiv:2206.00606.
        https://arxiv.org/pdf/2206.00606v3.pdf

    Parameters
    ----------
    in_channels : list of int
        Dimension of input features on vertices (0-cells), edges (1-cells) and faces (2-cells). The length of the list
        must be 3.
    intermediate_channels : list of int
        Dimension of intermediate features on vertices (0-cells), edges (1-cells) and faces (2-cells). The length of the
        list must be 3. The intermediate features are the ones computed after the first step of message passing.
    out_channels : list of int
        Dimension of output features on vertices (0-cells), edges (1-cells) and faces (2-cells). The length of the list
        must be 3. The output features are the ones computed after the second step of message passing.
    negative_slope : float
        Negative slope of LeakyReLU used to compute the attention coefficients.
    softmax_attention : bool, optional
        Whether to use softmax attention. If True, the attention coefficients are normalized by rows
        using softmax over all the columns that are not zero in the associated neighborhood matrix. If False,
        the normalization is done by dividing by the sum of the values of the coefficients in its row
        whose columns are not zero in the associated neighborhood matrix. Default is False.
    update_func_attention : callable, optional
        Activation function used in the attention block. If None, no activation function is applied. Default is None.
    update_func_aggregation : callable, optional
        Function used to aggregate the messages computed in each attention block. If None, the messages are aggregated
        by summing them. Default is None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of the attention layers. Default is 'xavier_uniform'.
    """

    def __init__(self, in_channels: List[int], intermediate_channels: List[int], out_channels: List[int],
                 negative_slope: float, softmax_attention=False, update_func_attention=None,
                 update_func_aggregation=None, initialization="xavier_uniform"):
        super().__init__()

        assert len(in_channels) == 3 and len(intermediate_channels) == 3 and len(out_channels) == 3

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        intermediate_channels_0, intermediate_channels_1, intermediate_channels_2 = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.hbs_0_level1 = HBS(source_in_channels=in_channels_0, source_out_channels=intermediate_channels_0,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_0_1_level1 = HBNS(source_in_channels=in_channels_1, source_out_channels=intermediate_channels_1,
                                    target_in_channels=in_channels_0, target_out_channels=intermediate_channels_0,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbns_1_2_level1 = HBNS(source_in_channels=in_channels_2, source_out_channels=intermediate_channels_2,
                                    target_in_channels=in_channels_1, target_out_channels=intermediate_channels_1,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbs_0_level2 = HBS(source_in_channels=intermediate_channels_0, source_out_channels=out_channels_0,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_0_1_level2 = HBNS(source_in_channels=intermediate_channels_1, source_out_channels=out_channels_1,
                                    target_in_channels=intermediate_channels_0, target_out_channels=out_channels_0,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbs_1_level2 = HBS(source_in_channels=intermediate_channels_1, source_out_channels=out_channels_1,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_1_2_level2 = HBNS(source_in_channels=intermediate_channels_2, source_out_channels=out_channels_2,
                                    target_in_channels=intermediate_channels_1, target_out_channels=out_channels_1,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbs_2_level2 = HBS(source_in_channels=intermediate_channels_2, source_out_channels=out_channels_2,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(self, x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2):
        r"""Forward pass.

        The forward pass of the Combinatorial Complex Attention Neural Network for Mesh Classification proposed
        in [HAJIJ23]_, Figure 35(b). The input features are transformed in two steps. In the first step, the
        intermediate features are computed using incidence and adjacency matrices. In the second step, the
        output features are computed using incidence, adjacency, and coadjacency matrices. The notation used
        in the code follows the one used in [PSHM23]_.

        References
        ----------
        .. [HAJIJ23] Mustafa Hajij et al. Topological Deep Learning: Going Beyond Graph Data.
            arXiv:2206.00606.
            https://arxiv.org/pdf/2206.00606v3.pdf

        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

            Parameters
            ----------
            x_0 : torch.Tensor, shape=[n_0_cells, in_channels[0]]
                Input features on the 0-cells (vertices) of the combinatorial complex.
            x_1 : torch.Tensor, shape=[n_1_cells, in_channels[1]]
                Input features on the 1-cells (edges) of the combinatorial complex.
            x_2 : torch.Tensor, shape=[n_2_cells, in_channels[2]]
            Input features on the 2-cells (faces) of the combinatorial complex.
            adjacency_0 : torch.sparse
                shape=[n_0_cells, n_0_cells]
                Neighborhood matrix mapping 0-cells to 0-cells (A_0_up).
            adjacency_1 : torch.sparse
                shape=[n_1_cells, n_1_cells]
                Neighborhood matrix mapping nodes to nodes (A_1_up).
            coadjacency_2 : torch.sparse
                shape=[n_2_cells, n_2_cells]
                Neighborhood matrix mapping nodes to nodes (A_2_down).
            incidence_1 : torch.sparse
                shape=[n_0_cells, n_1_cells]
                Neighborhood matrix mapping 1-cells to 0-cells (B_1).
            incidence_2 : torch.sparse
            shape=[n_1_cells, n_2_cells]
            Neighborhood matrix mapping 2-cells to 1-cells (B_2).

            Returns
            -------
            _ : torch.Tensor, shape=[1, num_classes]
                Output prediction on the entire cell complex.
            """
        # Computing messages from Higher Order Attention Blocks Level 1
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_1)
        x_1_to_2, x_2_to_1 = self.hbns_1_2_level1(x_2, x_1, incidence_2)

        x_0_level1 = self.aggr([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr([x_0_to_1, x_2_to_1])
        x_2_level1 = x_1_to_2

        # Computing messages from Higher Order Attention Blocks Level 2
        x_0_to_0 = self.hbs_0_level2(x_0_level1, adjacency_0)
        x_1_to_1 = self.hbs_1_level2(x_1_level1, adjacency_1)
        x_2_to_2 = self.hbs_2_level2(x_2_level1, coadjacency_2)

        x_0_to_1, _ = self.hbns_0_1_level2(x_1_level1, x_0_level1, incidence_1)
        x_1_to_2, _ = self.hbns_1_2_level2(x_2_level1, x_1_level1, incidence_2)

        x_0_level2 = x_0_to_0
        x_1_level2 = self.aggr([x_0_to_1, x_1_to_1])
        x_2_level2 = self.aggr([x_1_to_2, x_2_to_2])

        return x_0_level2, x_1_level2, x_2_level2
