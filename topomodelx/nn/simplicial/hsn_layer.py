"""High Skip Network Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class HSNLayer(torch.nn.Module):
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

        self.aggr_on_nodes = Aggregation(aggr_func="sum", update_func="sigmoid")

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
            m_{z \rightarrow x}^{(0 \rightarrow 0) \in \text{seq1}}
                = (A_{\uparrow,0})_{xz} \cdot m_{y \rightarrow z}^{(0 \rightarrow 0)\in \text{seq1}} \cdot \Theta^{t,1}

            m_{y \rightarrow z}^{(0 \rightarrow 1) \in \text{seq2}}
                = (B_1^T)_{zy} \cdot h_y^{t,(0)} \cdot \Theta^{t,0}

            m_{z \rightarrow u}^{(1 \rightarrow 1)1 \in \text{seq2}}
                = \sigma((A{\uparrow,1})_{uz} \cdot m_{y \rightarrow z}^{(0 \rightarrow 1) \in \text{seq2}} \cdot \Theta^{t,i,1})

            m_{u \rightarrow v}^{(1 \rightarrow 1)^i_2 \in \text{seq2}}
                = \sigma((A_{\uparrow,1})_{vu} \cdot m_{z \rightarrow u}^{(1 \rightarrow 1) \in \text{seq2}} \cdot \Theta_{t,i,1})

            m_{v \rightarrow w}^{(1 \rightarrow 1)^i_2 \in \text{seq2}}
                = \sigma((A_{\uparrow,0})_{wv} \cdot m_{u \rightarrow v}^{(1 \rightarrow 1)^i_2 \in \text{seq2}} \cdot \Theta_{t,i,2})

            m_{w \rightarrow s}^{(1 \rightarrow 0)\in \text{seq2}}
                = (B_1)_{sw} \cdot m_{v \rightarrow w}^{(1 \rightarrow 1)\_2^d \in \text{seq2}} \cdot \Theta^t

            m_{s \rightarrow x}^{(0 \rightarrow 0) \in \text{seq2}}
                = (A_{\uparrow,0})_{xs} \cdot m_{w \rightarrow s}^{(1 \rightarrow 0)\in \text{seq2}} \cdot \Theta^t

            m_{y \rightarrow z}^{(0 \rightarrow 1) \in \text{seq3}}
                = (B_1^T)_{zy} \cdot h_y^{t,(0)} \cdot \Theta^{t,0}

            m_{z \rightarrow z}^{(1)^i \in \text{seq3}}
                = m_{z \rightarrow z}^{(0,1) \text{ or } (1)^{i-1} \in \text{seq3}}

            m_{z \rightarrow w}^{(1 \rightarrow 0) \in \text{seq3}}
                = (B_1)_{wz} \cdot m_{y \rightarrow z}^{(1)^{d} \in \text{seq3}} \cdot \Theta^t

            m_{w \rightarrow x}^{(0 \rightarrow 0)\in \text{seq3}}
                = (A_{\uparrow,0})_{xw} \cdot m_{z \rightarrow w}^{(1 \rightarrow 0) \in \text{seq3}} \cdot \Theta^t

            m_{x}^{\text{seq1},(0)}
                = \sum_{z \in \mathcal{L}_\uparrow(x)} m_{z \rightarrow x}^{(0 \rightarrow 0) \in \text{seq1}}$

            m_{x}^{\text{seq2},(0)}
                = \sum_{s \in \mathcal{L}_\uparrow(x)} m_{s \rightarrow x}^{(0 \rightarrow 0) \in \text{seq2}}$

            m_{x}^{\text{seq3},(0)}
                = \sum_{w \in \mathcal{L}_\uparrow(x)} m_{w \rightarrow x}^{(0 \rightarrow 0) \in \text{seq3}}$

            m_x^{(0)}
                = m_x^{\text{seq1},(0)} + m_x^{\text{seq2},(0)} + m_x^{\text{seq3},(0)}$

            h_x^{t+1,(0)} = I(m_x^{(0)})$

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
        incidence_1_transpose = incidence_1.to_dense().T.to_sparse()

        x_0_level1 = self.conv_level1_0_to_0(x_0, adjacency_0)
        x_1_level1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)

        x_0_level2 = self.conv_level2_0_to_0(x_0_level1, adjacency_0)
        x_1_level2 = self.conv_level2_1_to_0(x_1_level1, incidence_1)

        x_0 = self.aggr_on_nodes([x_0_level2, x_1_level2])
        return x_0
