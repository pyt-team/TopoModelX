"""Simplical Complex Network Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCoNeLayer(torch.nn.Module):
    r"""Layer of a Simplical Complex Network (SCoNe).

    Implementation of this layer proposed in [RGS21]_.

    Notes
    -----
    This is the architecture proposed for trajectory prediction in [RGS21]_ repurposed for edge classification.

    References
    ----------
    .. [RGS21] Roddenberry, Glaze, Segarra.
        Principled Simplical Neural Networks for Trajectory Prediction.
        https://arxiv.org/pdf/2102.10058.pdf

    Parameters
    ----------
    channels : int
        Dimension of features on each simplicial cell.
    initialization : string
        Initialization method.
    """

    def __init__(self, channels):
        super().__init__()
        self.channels = channels

        self.conv_level1 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func=None,
        )

        self.conv_level2 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func=None,
        )

        self.conv_level3 = Conv(
            in_channels=channels,
            out_channels=channels,
            update_func="sigmoid",
        )

        self.aggr_edges = Aggregation(aggr_func="sum", update_func="sigmoid")

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1.reset_parameters()
        self.conv_level2.reset_parameters()
        self.conv_level3.reset_parameters()

    def forward(self, x_0, lap_up, lap_down, iden):
        r"""Forward pass.

        The forward pass was initially proposes in [RGS21]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        .. math::
            \begin{align*}
            🟥 $\quad m_{{y \rightarrow z \rightarrow x}}^{(1 \rightarrow 2 \rightarrow 1)} = \sigma ((L_{\uparrow,1})_{xy} \cdot h^{t,(1)}_y \cdot \Theta^{t,(1)1})$
            🟥 $\quad m_{y \rightarrow z \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}  = (L_{\downarrow,1})_{xy} \cdot h^{t, (1)}_y \cdot \Theta^{t,(1)2}$
            🟥 $\quad m_{{x \rightarrow x}}^{(1 \rightarrow 1)}  = h_x^{t,(1)} \cdot \Theta^{t,(1)3}$
            🟧 $\quad m^{(1 \rightarrow 2 \rightarrow 1)} = \sum_{y \in \mathcal{L}_\uparrow(x)} m_{{y \rightarrow x}}^{(1 \rightarrow 2 \rightarrow 1)}$
            🟧 $\quad m^{(1 \rightarrow 0 \rightarrow 1)}  = \sum_{y \in \mathcal{L}_\downarrow(x)} m_{y \rightarrow z \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}$
            🟩 $\quad m_x^{(1)}  = m_x^{(1 \rightarrow 2 \rightarrow 1)} + m_x^{(1 \rightarrow 0 \rightarrow 1)} + m_{x \rightarrow x}^{1 \rightarrow 1}$
            🟦 $\quad h_x^{t+1,(1)}  = \sigma(m_x^{(1)})$
            \end{align*}

        References
        ----------
        .. [RGS21] Roddenberry, Glaze, Segarra.
            Principled Simplical Neural Networks for Trajectory Prediction.
            https://arxiv.org/pdf/2102.10058.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0: torch.Tensor, shape=[n_edges, channels]
            Input features on the edges of the simplical complex.
        lap_up: torch.sparse, shape=[n_edges, n_edges]
            Laplacian matrix (up) :math: 'L_{\uparrow,1}' mapping edges to faces then back.
        lap_down: torch.sparse, shape=[n_edges, n_edges]
            Laplacian matrix (down) :math: 'L_{\downarrow, 1}' mapping edges to nodes then back.
        iden: torch.sparse, shape=[n_edges, n_edges]
            Identity matrix simply keeping values the same to be acted on by learnable parameters

        Returns
        -------
        _ : torch.Tensor, shape=[n_edges, channels]
            Output features on the edges of the simplical complex.
        """
        x_0_level1 = self.conv_level1(x_0, lap_down)
        x_0_level2 = self.conv_level2(x_0, lap_up)
        x_0_level3 = self.conv_level3(x_0, iden)
        x_0 = self.aggr_edges([x_0_level3, x_0_level1, x_0_level2])
        return x_0
