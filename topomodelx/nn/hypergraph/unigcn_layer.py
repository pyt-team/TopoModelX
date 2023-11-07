"""Implementation of UniGCN layer from Huang et. al.: UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks."""
import torch
from torch import nn

from topomodelx.base.conv import Conv


class UniGCNLayer(torch.nn.Module):
    """Layer of UniGCN.

    Implementation of UniGCN layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    use_bn : boolean, default=False
        Whether to use bathnorm after the linear transformation.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        aggr_norm: bool = False,
        use_bn: bool = False,
    ) -> None:
        super().__init__()

        with_linear_transform = False if in_channels == hidden_channels else True
        self.conv_level1_0_to_1 = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggr_norm=aggr_norm,
            update_func=None,
            with_linear_transform=with_linear_transform,
        )
        self.conv_level2_1_to_0 = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr_norm=aggr_norm,
            update_func=None,
        )
        self.bn = nn.BatchNorm1d(hidden_channels) if use_bn else None

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.conv_level1_0_to_1.reset_parameters()
        self.conv_level2_1_to_0.reset_parameters()
        if self.bn is not None:
            self.bn.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""[1]_ initially proposed the forward pass.

        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of three steps.

        First, every hyper-edge sums up the features of its constituent edges:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)}  = B_1^T \cdot h_y^{t, (0)}\\
            &🟧 \quad m_z^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\\
            \end{align*}

        Second, the message to the nodes is the sum of the messages from the incident hyper-edges:

        .. math::
            \begin{align*}
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)} = B_1^{t,(1)} \cdot w^{(1)} \cdot  m_z^{(0 \rightarrow 1)} \cdot \Theta^t\\
            &🟧 \quad m_x^{(1 \rightarrow 0)}  = \sum_{y \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1 \rightarrow 0)}\\
            \end{align*}

        Third, the node features are updated:

        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_x^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = m_x^{(0)}
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels)
            Input features on the nodes of the hypergraph.
        incidence_1 : torch.sparse, shape = (n_nodes, n_edges)
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        if x_0.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Mismatch in number of nodes in features and nodes: {x_0.shape[-2]} and {incidence_1.shape[-2]}."
            )

        incidence_1_transpose = incidence_1.transpose(1, 0)
        x_1 = self.conv_level1_0_to_1(x_0, incidence_1_transpose)
        if self.bn is not None:
            x_1 = self.bn(x_1)
        x_0 = self.conv_level2_1_to_0(x_1, incidence_1)
        return (x_0, x_1)
