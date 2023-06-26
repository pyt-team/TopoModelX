"""UniGAT layer implementation."""
import torch

from topomodelx.base.conv import Conv


class UniGATLayer(torch.nn.Module):
    r"""Implementation of the UniGAT layer.

    References
    ----------
    .. [JJ21]Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
        In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
        2021. https://arxiv.org/pdf/2105.00956.pdf

    Parameters
    ----------
    in_channels : int
        Number of input channels on node features.
    out_channels : int
        Number of output channels.
    """

    def __init__(self, in_channels, out_channels, aggr_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

        self.conv_1_0 = Conv(in_channels, out_channels, aggr_norm=aggr_norm, att=True)

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_1_0.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""Forward pass proposed in [JJ21]_.

        The forward pass of the UniGAT layer is defined as:

        1. Every hyper-edge sums up the features of its constituent edges:
        .. math::
            \begin{align*}
            &:red_square: \quad m_{y \rightarrow z}^{(0 \rightarrow 1)} = (B^T_1)\_{zy} \cdot h^{t,(0)}_y \\
            &:orange_square: \quad m_z^{(0\rightarrow1)} = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}
            \end{align*}

        2. The message to the nodes is computed using self-attention:
        .. math::
            \begin{align*}
            &:red_square: \quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = ((B_1 \odot att(h_{z \in \mathcal{C}(x)}^{t,(1)})))\_{xz} \cdot m_{z}^{(0\rightarrow1)} \cdot \Theta^{t,(1)} \\
            &:orange_square: \quad m_{x}^{(1 \rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)}
            \end{align*}

        3. The node features are updated:
        .. math::
            \begin{align*}
            &:green_square: \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &:blue_square: \quad h_x^{t+1,(0)}  = m_x^{(0)}
            \end{align*}

        References
        ----------
        .. [JJ21] Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
            In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
            2021. https://arxiv.org/pdf/2105.00956.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, in_channels]
            Input features on the nodes of the hypergraph.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_0 : torch.Tensor, shape=[n_nodes, out_channels]
            Output features on the nodes of the hypergraph.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)

        # first message passing step without learning / parameters
        m_0_1 = torch.sparse.mm(incidence_1_transpose.float(), x_0)
        # second message passing step using attention
        # TODO: implement correct attention mechanism
        m_1_0 = self.conv_1_0(m_0_1, incidence_1, x_0)

        # final update steps are identity mappings
        return m_1_0
