"""Simplicial Complex Net Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCoNeLayer(torch.nn.Module):
    """Layer of a High Skip Network (HSN).

    Implementation of the SCoNe layer proposed in [RGS21]_.

    Notes
    -----
    This is the architecture proposed for trajectory prediction on simplicial complexes. 

    For the trajectory prediction architecture proposed in [RGS21]_, these layers are stacked before applying the boundary map from 1-chains to 0-chains. Finally, one can apply the softmax operator on the neighbouring nodes of the last node in the given trajectory to predict the next node. When implemented like this, we get a map from (ordered) 1-chains (trajectories/paths) to the neighbouring nodes of the last node in the 1-chain.

    References
    ----------
	.. [RGS21] Roddenberry, Mitchell, Glaze.
		Principled Simplicial Neural Networks for Trajectory Prediction.
		Proceedings of the 38th International Conference on Machine Learning.
		https://proceedings.mlr.press/v139/roddenberry21a.html

    Parameters
    ----------
    in_channels : int
        Input dimension of features on each edge.
    out_channels : int
        Output dimension of features on each edge.
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        pass

    def forward(self, x, incidence_1, incidence_2):
        r"""Forward pass.

        The forward pass was initially proposed in [RGS21]_.
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
        .. [RGS21] Roddenberry, Mitchell, Glaze.
            Principled Simplicial Neural Networks for Trajectory Prediction.
            Proceedings of the 38th International Conference on Machine Learning.
            https://proceedings.mlr.press/v139/roddenberry21a.html
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x: torch.Tensor, shape=[n_edges, in_channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse, shape=[n_nodes, n_edges]
            Incidence matrix :math:`B_1` mapping edges to nodes.
        incidence_2 : torch.sparse, shape=[n_edges, n_triangles]
            Incidence matrix :math:`B_2` mapping triangles to edges.

        Returns
        -------
        _ : torch.Tensor, shape=[n_edges, out_channels]
            Output features on the edges of the simplicial complex.
        """
        pass
