"""High Skip Network Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class Dist2CycleLayer(torch.nn.Module):
    """Layer of a High Skip Network (HSN).

    Implementation of the HSN layer proposed in [HRGZ22]_.

    Notes
    -----
    This is the architecture proposed for node classification on simplicial complices.

    References
    ----------
    .. [HRGZ22] Hajij, Ramamurthy, Guzmán-Sáenz, Zamzmi.
        High Skip Networks: A Higher Order Generalization of Skip Connections.
        Geometrical and Topological Representation Learning Workshop at ICLR 2022.
        https://openreview.net/pdf?id=Sc8glB-k6e9

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

        self.aggr_on_nodes = Aggregation(
            aggr_func="sum", update_func="sigmoid")

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_0_to_0.reset_parameters()
        self.conv_level1_0_to_1.reset_parameters()
        self.conv_level2_0_to_0.reset_parameters()
        self.conv_level2_1_to_0.reset_parameters()

    def forward(self, x_0, incidence_1, adjacency_0):
        r"""Forward pass.

        The forward pass was initially proposed in [HRGZ22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        .. math::
            \begin{align*}
            &🟥 \quad m_{{y \rightarrow z}}^{(0 \rightarrow 0)} = \sigma ((A_{\uparrow,0})_{xy} \cdot h^{t,(0)}_y \cdot \Theta^{t,(0)1})\\
            &🟥 \quad m_{z \rightarrow x}^{(0 \rightarrow 0)}  = (A_{\uparrow,0})_{xy} \cdot m_{y \rightarrow z}^{(0 \rightarrow 0)} \cdot \Theta^{t,(0)2}\\
            &🟥 \quad m_{{y \rightarrow z}}^{(0 \rightarrow 1)}  = \sigma((B_1^T)_{zy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow 1)})\\
            &🟥 \quad m_{z \rightarrow x)}^{(1 \rightarrow 0)}  = (B_1)_{xz} \cdot m_{z \rightarrow x}^{(0 \rightarrow 1)} \cdot \Theta^{t, (1 \rightarrow 0)}\\
            &🟧 \quad m_{x}^{(0 \rightarrow 0)}  = \sum_{z \in \mathcal{L}_\uparrow(x)} m_{z \rightarrow x}^{(0 \rightarrow 0)}\\
            &🟧 \quad m_{x}^{(1 \rightarrow 0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1 \rightarrow 0)}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(0 \rightarrow 0)} + m_x^{(1 \rightarrow 0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = I(m_x^{(0)})
            \end{align*}

        References
        ----------
        .. [HRGZ22] Hajij, Ramamurthy, Guzmán-Sáenz, Zamzmi.
            High Skip Networks: A Higher Order Generalization of Skip Connections.
            Geometrical and Topological Representation Learning Workshop at ICLR 2022.
            https://openreview.net/pdf?id=Sc8glB-k6e9
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x: torch.Tensor, shape=[n_nodes, channels]
            Input features on the nodes of the simplicial complex.
        incidence_1 : torch.sparse, shape=[n_nodes, n_edges]
            Incidence matrix :math:`B_1` mapping edges to nodes.
        adjacency_0 : torch.sparse, shape=[n_nodes, n_nodes]
            Adjacency matrix :math:`A_0^{\uparrow}` mapping nodes to nodes via edges.

        Returns
        -------
        _ : torch.Tensor, shape=[n_nodes, channels]
            Output features on the nodes of the simplicial complex.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)

        x_0_level1 = self.conv_level1_0_to_0(x_0, adjacency_0)
        x_1_level1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)

        x_0_level2 = self.conv_level2_0_to_0(x_0_level1, adjacency_0)
        x_1_level2 = self.conv_level2_1_to_0(x_1_level1, incidence_1)

        x_0 = self.aggr_on_nodes([x_0_level2, x_1_level2])
        return x_0
