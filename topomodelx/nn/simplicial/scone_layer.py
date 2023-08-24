"""Simplicial Complex Net Layer."""
import torch

from topomodelx.base.aggregation import Aggregation


class SCoNeLayer(torch.nn.Module):
    """
    Implementation of the SCoNe layer proposed in [RGS21]_.

    Notes
    -----
    This is the architecture proposed for trajectory prediction on simplicial complexes.

    For the trajectory prediction architecture proposed in [RGS21]_, these layers are stacked before applying the boundary map from 1-chains to 0-chains. Finally, one can apply the softmax operator on the neighbouring nodes of the last node in the given trajectory to predict the next node. When implemented like this, we get a map from (ordered) 1-chains (trajectories) to the neighbouring nodes of the last node in the 1-chain.

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
    update_func : string
        Update function to use when updating edge features.
    """

    def __init__(
        self, in_channels: int, out_channels: int, update_func: str = "tanh"
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.weight_0 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight_1 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight_2 = torch.nn.parameter.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.aggr_on_edges = Aggregation("sum", update_func)

    def reset_parameters(self, gain: float = 1.0):
        torch.nn.init.xavier_uniform_(self.weight_0, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_1, gain=gain)
        torch.nn.init.xavier_uniform_(self.weight_2, gain=gain)

    def forward(
        self, x: torch.Tensor, incidence_1: torch.Tensor, incidence_2: torch.Tensor
    ) -> torch.Tensor:
        r"""Forward pass.

        The forward pass was initially proposed in [RGS21]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        .. math::
            \begin{align*}
            &🟥 \quad m^{(1 \rightarrow 0 \rightarrow 1)}_{y \rightarrow \{z\} \rightarrow x}  = (L_{\downarrow,1})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟥 \quad m_{x \rightarrow x}^{(1 \rightarrow 1)}  = h_x^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 1)}\\
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}  = (L_{\uparrow,1})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 2 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1 \rightarrow 0 \rightarrow 1)} = \sum_{y \in \mathcal{L}_\downarrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1 \rightarrow 2 \rightarrow 1)}  = \sum_{y \in \mathcal{L}_\uparrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}\\
            &🟩 \quad m_x^{(1)}  = m_{x}^{(1 \rightarrow 0 \rightarrow 1)} + m_{x \rightarrow x}^{(1 \rightarrow 1)} + m_{x}^{(1 \rightarrow 2 \rightarrow 1)}\\
            &🟦 \quad h_x^{t,(1)} = \sigma(m_x^{(1)})
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
        z1 = incidence_2 @ incidence_2.T @ x @ self.weight_2
        z2 = x @ self.weight_1
        z3 = incidence_1.T @ incidence_1 @ x @ self.weight_0
        out = self.aggr_on_edges([z1, z2, z3])
        return out
