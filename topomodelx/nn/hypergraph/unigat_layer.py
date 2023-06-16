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
        Number of input channels.
    """

    def __init__(self, in_channels, out_channels, aggr_norm=False):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        pass

    def forward(self, x_0, incidence_1):
        r"""Forward pass proposed in [JJ21]_.

        The forward pass of the UniGAT layer is defined as:

        1. Every hyper-edge sums up the features of its constituent edges:

        2. The message to the nodes is the sum of the messages from the incident hyper-edges:

        3. The node features are updated:
        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = m_x^{(0)}
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
        # incidence_1_transpose = incidence_1.transpose(1, 0)

        return x_0
