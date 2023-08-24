"""UniGCNII layer implementation."""
import torch


class UniGCNIILayer(torch.nn.Module):
    r"""Implementation of the UniGCNII layer.

    Parameters
    ----------
    in_channels : int
        Number of input channels of the node (rank-0) features.
    alpha : float
        The alpha parameter determining the importance of the self-loop (\theta_2).
    beta : float
        The beta parameter determining the importance of the learned matrix (\theta_1).

    References
    ----------
    ..  [JJ21] Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
        In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
        2021. https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(self, in_channels, alpha, beta):
        super().__init__()

        self.in_channels = in_channels
        self.alpha = alpha
        self.beta = beta
        self.linear = torch.nn.Linear(in_channels, in_channels, bias=False)

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.linear.reset_parameters()

    def forward(self, x_0, incidence_1, x_skip=None):
        r"""Forward pass of the UniGCNII layer.

        The forward pass consists of:
        - two messages, and
        - a skip connection with a learned update function.

        1. Every hyper-edge sums up the features of its constituent edges:
        .. math::
            \begin{align*}
            & 🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)} = (B^T_1)\_{zy} \cdot h^{t,(0)}_y \\
            & 🟧 \quad m_z^{(0\rightarrow1)} = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}
            \end{align*}

        2. The second message is normalized with the node and edge degrees:
        .. math::
            \begin{align*}
            & 🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)} \\
            & 🟧 \quad m_{x}^{(1\rightarrow0)}  = \frac{1}{\sqrt{d_x}}\sum_{z \in \mathcal{C}(x)} \frac{1}{\sqrt{d_z}}m_{z \rightarrow x}^{(1\rightarrow0)} \\
            \end{align*}

        3. The computed message is combined with skip connections and a linear transformation using hyperparameters alpha and beta:
        .. math::
            \begin{align*}
            & 🟩 \quad m_x^{(0)}  = m_x^{(1 \rightarrow 0)} \\
            & 🟦 \quad m_x^{(0)}  = ((1-\beta)I + \beta W)((1-\alpha)m_x^{(0)} + \alpha \cdot h_x^{t,(0)}) \\
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = [num_nodes, in_channels]
            Input features of the nodes of the hypergraph.
        incidence_1 : torch.Tensor, shape = [num_nodes, num_edges]
            Incidence matrix of the hypergraph.
            It is expected that the incidence matrix contains self-loops for all nodes.
        x_skip : torch.Tensor, shape = [num_nodes, in_channels], optional
            Original node features of the hypergraph used for the skip connections.
            If not provided, the input to the layer is used as a skip connection.

        Returns
        -------
        x_0 : torch.Tensor, shape = [num_nodes, in_channels]
            Output features of the nodes of the hypergraph.

        """
        x_skip = x_0 if x_skip is None else x_skip
        incidence_1_transpose = incidence_1.transpose(0, 1)

        # First message without any learning or parameters
        m_0_1 = torch.sparse.mm(incidence_1_transpose, x_0)

        # Compute node and edge degrees for normalization.
        node_degree = torch.sum(incidence_1.to_dense(), dim=1)

        # check if the node degrees are positive
        assert torch.all(
            node_degree > 0
        ), "Node degrees should be positive (at least self-loops should be included).)"

        # Average node degree for each edge.
        edge_degree = torch.sum(torch.diag(node_degree) @ incidence_1, dim=0)
        assert torch.all(
            edge_degree > 0
        ), "Edge degrees should be positive (every edge needs at least one node it is connecting)."
        edge_degree = edge_degree / torch.sum(incidence_1.to_dense(), dim=0)

        # Second message normalized with node and edge degrees (using broadcasting)
        m_1_0 = (1 / torch.sqrt(node_degree).unsqueeze(-1)) * torch.sparse.mm(
            incidence_1 @ torch.diag(1 / torch.sqrt(edge_degree)), m_0_1
        )

        # Introduce skip connections with hyperparameter alpha and beta
        x_combined = ((1 - self.alpha) * m_1_0) + (self.alpha * x_skip)
        return ((1 - self.beta) * x_combined) + self.beta * self.linear(x_combined)
