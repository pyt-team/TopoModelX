"""Simplicial 2-complex convolutional neural network."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCConvLayer(torch.nn.Module):
    """Layer of a Simplicial 2-complex convolutional neural network (SCConv).

    Implementation of the SCConv layer proposed in [1]_.

    References
    ----------
    .. [1] Bunch, You, Fung and Singh.
        Simplicial 2-complex convolutional neural nets.
        TDA and beyond, NeurIPS 2020 workshop.
        https://openreview.net/forum?id=TLbnsKrt6J-
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(self, node_channels, edge_channels, face_channels) -> None:
        super().__init__()

        self.node_channels = node_channels
        self.edge_channels = edge_channels
        self.face_channels = face_channels

        self.conv_0_to_0 = Conv(
            in_channels=self.node_channels,
            out_channels=self.node_channels,
            update_func=None,
        )
        self.conv_0_to_1 = Conv(
            in_channels=self.node_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )

        self.conv_1_to_1 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )
        self.conv_1_to_0 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.node_channels,
            update_func=None,
        )

        self.conv_1_to_2 = Conv(
            in_channels=self.edge_channels,
            out_channels=self.face_channels,
            update_func=None,
        )

        self.conv_2_to_1 = Conv(
            in_channels=self.face_channels,
            out_channels=self.edge_channels,
            update_func=None,
        )

        self.conv_2_to_2 = Conv(
            in_channels=self.face_channels,
            out_channels=self.face_channels,
            update_func=None,
        )

        self.aggr_on_nodes = Aggregation(aggr_func="sum", update_func="sigmoid")
        self.aggr_on_edges = Aggregation(aggr_func="sum", update_func="sigmoid")
        self.aggr_on_faces = Aggregation(aggr_func="sum", update_func="sigmoid")

    def reset_parameters(self) -> None:
        r"""Reset parameters."""
        self.conv_0_to_0.reset_parameters()
        self.conv_0_to_1.reset_parameters()
        self.conv_1_to_0.reset_parameters()
        self.conv_1_to_1.reset_parameters()
        self.conv_1_to_2.reset_parameters()
        self.conv_2_to_1.reset_parameters()
        self.conv_2_to_2.reset_parameters()

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
        r"""Forward pass (see [2]_ and [3]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y\rightarrow x}^{(0\rightarrow 0)} = ({\tilde{A}_{\uparrow,0}})_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0\rightarrow0)}\\
            &🟥 \quad m^{(1\rightarrow0)}_{y\rightarrow x}  = (B_1)_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(1\rightarrow 0)}\\
            &🟥 \quad m^{(0 \rightarrow 1)}_{y \rightarrow x}  = (\tilde B_1)_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow1)}\\
            &🟥 \quad m^{(1\rightarrow1)}_{y\rightarrow x} = ({\tilde{A}_{\downarrow,1}} + {\tilde{A}_{\uparrow,1}})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1\rightarrow1)}\\
            &🟥 \quad m^{(2\rightarrow1)}_{y \rightarrow x}  = (B_2)_{xy} \cdot h_y^{t,(2)} \cdot \Theta^{t,(2 \rightarrow1)}\\
            &🟥 \quad m^{(1 \rightarrow 2)}_{y \rightarrow x}  = (\tilde B_2)_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 2)}\\
            &🟥 \quad m^{(2 \rightarrow 2)}_{y \rightarrow x}  = ({\tilde{A}_{\downarrow,2}})\_{xy} \cdot h_y^{t,(2)} \cdot \Theta^{t,(2 \rightarrow 2)}\\
            &🟧 \quad m_x^{(0 \rightarrow 0)}  = \sum_{y \in \mathcal{L}_\uparrow(x)} m_{y \rightarrow x}^{(0 \rightarrow 0)}\\
            &🟧 \quad m_x^{(1 \rightarrow 0)}  = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(1 \rightarrow 0)}\\
            &🟧 \quad m_x^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(0 \rightarrow 1)}\\
            &🟧 \quad m_x^{(1 \rightarrow 1)}  = \sum_{y \in (\mathcal{L}_\uparrow(x) + \mathcal{L}_\downarrow(x))} m_{y \rightarrow x}^{(1 \rightarrow 1)}\\
            &🟧 \quad m_x^{(2 \rightarrow 1)} = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(2 \rightarrow 1)}\\
            &🟧 \quad m_x^{(1 \rightarrow 2)}  = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(1 \rightarrow 2)}\\
            &🟧 \quad m_x^{(2 \rightarrow 2)}  = \sum_{y \in \mathcal{L}_\downarrow(x)} m_{y \rightarrow x}^{(2 \rightarrow 2)}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}+ m_x^{(0\rightarrow0)}\\
            &🟩 \quad m_x^{(1)}  = m_x^{(2\rightarrow1)}+ m_x^{(1\rightarrow1)}\\
            &🟦 \quad h^{t+1, (0)}_x  = \sigma(m_x^{(0)})\\
            &🟦 \quad h^{t+1, (1)}_x  = \sigma(m_x^{(1)})\\
            &🟦 \quad h^{t+1, (2)}_x  = \sigma(m_x^{(2)})
            \end{align*}

        Parameters
        ----------
        x_0: torch.Tensor, shape=[n_nodes, node_channels]
            Input features on the nodes of the simplicial complex.
        x_1: torch.Tensor, shape=[n_edges, edge_channels]
            Input features on the edges of the simplicial complex.
        x_2: torch.Tensor, shape=[n_faces, face_channels]
            Input features on the faces of the simplicial complex.
        incidence_1: torch.Tensor, shape=[n_faces, channels]
            incidence matrix of rank 1 :math:`B_1`.
        incidence_1_norm: torch.Tensor,
            normalized incidence matrix of rank 1 :math:`B^{~}_1`.
        incidence_2: torch.Tensor,
             incidence matrix of rank 2 :math:`B_2`.
        incidence_2_norm: torch.Tensor,
            normalized incidence matrix of rank 2 :math:`B^{~}_2`.
        adjacency_up_0_norm: torch.Tensor,
            normalized upper adjacency matrix of rank 0.
        adjacency_up_1_norm: torch.Tensor,
            normalized upper adjacency matrix of rank 1.
        adjacency_down_1_norm: torch.Tensor,
            normalized down adjacency matrix of rank 1.
        adjacency_down_2_norm: torch.Tensor,
            normalized down adjacency matrix of rank 2.

        Notes
        -----
        For normalization of incidence matrices you may use the helper functions here: https://github.com/pyt-team/TopoModelX/blob/dev/topomodelx/normalization/normalization.py

        """
        x0_level_0_0 = self.conv_0_to_0(x_0, adjacency_up_0_norm)

        x0_level_1_0 = self.conv_1_to_0(x_1, incidence_1)

        x0_level_0_1 = self.conv_0_to_1(x_0, incidence_1_norm)

        adj_norm = adjacency_down_1_norm.add(adjacency_up_1_norm)
        x1_level_1_1 = self.conv_1_to_1(x_1, adj_norm)

        x2_level_2_1 = self.conv_2_to_1(x_2, incidence_2)

        x1_level_1_2 = self.conv_1_to_2(x_1, incidence_2_norm)

        x_2_level_2_2 = self.conv_2_to_2(x_2, adjacency_down_2_norm)

        x0_out = self.aggr_on_nodes([x0_level_0_0, x0_level_1_0])
        x1_out = self.aggr_on_edges([x0_level_0_1, x1_level_1_1, x2_level_2_1])
        x2_out = self.aggr_on_faces([x1_level_1_2, x_2_level_2_2])

        return x0_out, x1_out, x2_out
