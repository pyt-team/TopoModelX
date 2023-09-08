"""Implementation of UniGCN layer from Huang et. al.: UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks."""
import torch
from torch import nn

from topomodelx.base.conv import Conv


class UniGCNLayer(torch.nn.Module):
    """Layer of UniGCN.

    Implementation of UniGCN layer proposed in [JJ21]_.


    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    use_bn : boolean
        Whether to use bathnorm after the linear transformation.
    aggr_norm: boolean
        Whether to normalize the aggregated message by the neighborhood size.

    References
    ----------
    ..  [JJ21]Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
        In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
        2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self, in_channels, out_channels, aggr_norm: bool = False, use_bn: bool = False
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_level1_0_to_1 = Conv(
            in_channels=in_channels,
            out_channels=in_channels,
            aggr_norm=aggr_norm,
            update_func=None,
            with_linear_transform=False,
        )
        self.conv_level2_1_to_0 = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr_norm=aggr_norm,
            update_func=None,
        )
        self.bn = nn.BatchNorm1d(in_channels) if use_bn else None

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.conv_level1_0_to_1.reset_parameters()
        self.conv_level2_1_to_0.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""[JJ21]_ initially proposed the forward pass.

        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        The forward pass of this layer is composed of three steps.

        1. Every hyper-edge sums up the features of its constituent edges:
        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)}  = B_1^T \cdot h_y^{t, (0)}\\
            &🟧 \quad m_z^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\\
            \end{align*}

        2. The message to the nodes is the sum of the messages from the incident hyper-edges:
        .. math::
            \begin{align*}
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = B_1^{t,(1)} \cdot w^{(1)} \cdot  m_z^{(0 \rightarrow 1)} \cdot \Theta^t\\
            &🟧 \quad m_x^{(1 \rightarrow 0)}  = \sum_{y \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1 \rightarrow 0)}\\
            \end{align*}

        3. The node features are updated:
        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = m_x^{(0)}
            \end{align*}

        References
        ----------
        ..  [JJ21]Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
            In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
            2021.
            https://arxiv.org/pdf/2105.00956.pdf
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
        if x_0.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Mismatch in number of nodes in features and nodes: {x_0.shape[-2]} and {incidence_1.shape[-2]}."
            )

        incidence_1_transpose = incidence_1.transpose(1, 0)
        m_0_1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)
        if self.bn is not None:
            m_0_1 = self.bn(m_0_1)
        m_1_0 = self.conv_level2_1_to_0(m_0_1, incidence_1)
        return m_1_0
