"""Simplicial 2-complex convolutional neural network."""
import torch


class SCConvLayer(torch.nn.Module):
    """Layer of a Simplicial 2-complex convolutional neural network (SCConv).

    Implementation of the SCConv layer proposed in [Bunch20]_.
    References
    ----------
    .. [Bunch20] Bunch, Eric, Qian You, Glenn Fung, and Vikas Singh.
        Simplicial 2-complex convolutional neural nets.
        NeurIPS 2020 Workshop TDA and Beyond homepage
        https://openreview.net/forum?id=TLbnsKrt6J-

    """

    def __init__(
        self,
        channels,
    ):
        super().__init__()

    def reset_parameters(self):
        r"""reset parameters."""

    #
    # def  normalize_adjacency(self, A):


    def forward(self,x_0,x_1,x_2):
        r"""Forward pass.

        .. math::
        \begin{align*}
        &🟥 \quad m_{y\rightarrow x}^{(0\rightarrow 0)} = ({\tilde{A}_{\uparrow,0}})_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0\rightarrow0)}
        &🟥 \quad m^{(1\rightarrow0)}_{y\rightarrow x}  = (B_1)_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(1\rightarrow 0)}
        &🟥 \quad m^{(0 \rightarrow 1)}_{y \rightarrow x}  = (\tilde B_1)_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0 \rightarrow1)}
        &🟥 \quad m^{(1\rightarrow1)}_{y\rightarrow x} = ({\tilde{A}_{\downarrow,1}} + {\tilde{A}_{\uparrow,1}})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1\rightarrow1)}
        &🟥 \quad m^{(2\rightarrow1)}_{y \rightarrow x}  = (B_2)_{xy} \cdot h_y^{t,(2)} \cdot \Theta^{t,(2 \rightarrow1)}
        &🟥 \quad m^{(1 \rightarrow 2)}_{y \rightarrow x}  = (\tilde B_2)_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 2)}
        &🟥 \quad m^{(2 \rightarrow 2)}_{y \rightarrow x}  = ({\tilde{A}_{\downarrow,2}})\_{xy} \cdot h_y^{t,(2)} \cdot \Theta^{t,(2 \rightarrow 2)}
        &🟧 \quad m_x^{(0 \rightarrow 0)}  = \sum_{y \in \mathcal{L}_\uparrow(x)} m_{y \rightarrow x}^{(0 \rightarrow 0)}
        &🟧 \quad m_x^{(1 \rightarrow 0)}  = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(1 \rightarrow 0)}
        &🟧 \quad m_x^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(0 \rightarrow 1)}
        &🟧 \quad m_x^{(1 \rightarrow 1)}  = \sum_{y \in (\mathcal{L}_\uparrow(x) + \mathcal{L}_\downarrow(x))} m_{y \rightarrow x}^{(1 \rightarrow 1)}
        &🟧 \quad m_x^{(2 \rightarrow 1)} = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(2 \rightarrow 1)}
        &🟧 \quad m_x^{(1 \rightarrow 2)}  = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(1 \rightarrow 2)}
        &🟧 \quad m_x^{(2 \rightarrow 2)}  = \sum_{y \in \mathcal{L}_\downarrow(x)} m_{y \rightarrow x}^{(2 \rightarrow 2)}
        &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}+ m_x^{(0\rightarrow0)}
        &🟩 \quad m_x^{(1)}  = m_x^{(2\rightarrow1)}+ m_x^{(1\rightarrow1)}
        &🟦 \quad h^{t+1, (0)}_x  = \sigma(m_x^{(0)})
        &🟦 \quad h^{t+1, (1)}_x  = \sigma(m_x^{(1)})
        &🟦 $\quad h^{t+1, (2)}_x  = \sigma(m_x^{(2)})
        \end{align*}

         References
        ----------
        .. [Bunch20] Bunch, Eric, Qian You, Glenn Fung, and Vikas Singh.
            Simplicial 2-complex convolutional neural nets.
            NeurIPS 2020 Workshop TDA and Beyond homepage
            https://openreview.net/forum?id=TLbnsKrt6J-
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0: torch.Tensor, shape=[n_nodes, channels]
            Input features on the nodes of the simplicial complex.
        x_1: torch.Tensor, shape=[n_edges, channels]
            Input features on the edges of the simplicial complex.
        x_2: torch.Tensor, shape=[n_faces, channels]
            Input features on the faces of the simplicial complex.

        """



