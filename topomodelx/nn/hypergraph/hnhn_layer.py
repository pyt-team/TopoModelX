"""Template Layer with two conv passing steps."""
import torch
from torch.nn.parameter import Parameter

from topomodelx.base.conv import Conv


class HNHNLayer(torch.nn.Module):
    """Layer of a Hypergraph Networks with Hyperedge Neurons (HNHN).

    Implementation of a simplified version of the HNHN layer proposed in [DSB20]_.

    This layer is composed of two convolutional layers:
    1. A convolutional layer sending messages from edges to nodes.
    2. A convolutional layer sending messages from nodes to edges.
    The incidence matrices can be normalized usign the node and edge cardinality.
    Two hyperparameters alpha and beta, control the normalization strenght.
    The convolutional layers support the training of a bias term.

    Notes
    -----
    This is the architecture proposed for node classification.

    References
    ----------
    .. [DSB20] Dong, Sawin, Bengio.
        HNHN: Hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020
        https://grlplus.github.io/papers/40.pdf


    Parameters
    ----------
    channels_node : int
        Dimension of node features.
    channels_edge : int
        Dimension of edge features.
    incidence_1 : torch.sparse
        Incidence matrix mapping edges to nodes (B_1).
        shape=[n_nodes, n_edges]
    use_bias : bool
        Flag controlling whether to use a bias term in the convolution.
    use_normalized_incidence : bool
        Flag controlling whether to normalize the incidence matrices.
    alpha : float
        Scalar controlling the importance of edge cardinality.
    beta : float
        Scalar controlling the importance of node cardinality.
    bias_gain : float
        Gain for the bias initialization.
    bias_init : string ["xavier_uniform"|"xavier_normal"]
        Controls the bias initialization method.
    """

    def __init__(
        self,
        channels_node,
        channels_edge,
        incidence_1,
        use_bias: bool = True,
        use_normalized_incidence: bool = True,
        alpha: float = -1.5,
        beta: float = -0.5,
        bias_gain: float = 1.414,
        bias_init: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        self.use_bias = use_bias
        self.bias_init = bias_init
        self.bias_gain = bias_gain
        self.use_normalized_incidence = use_normalized_incidence
        self.incidence_1 = incidence_1
        self.incidence_1_transpose = incidence_1.transpose(1, 0)
        self.channels_edge = channels_edge
        self.channels_node = channels_node
        self.conv_1_to_0 = Conv(
            in_channels=channels_edge,
            out_channels=channels_node,
            aggr_norm=False,
            update_func=None,
        )
        self.conv_0_to_1 = Conv(
            in_channels=channels_node,
            out_channels=channels_edge,
            aggr_norm=False,
            update_func=None,
        )
        if self.use_bias:
            self.bias_1_to_0 = Parameter(torch.Tensor(1, channels_node))
            self.bias_0_to_1 = Parameter(torch.Tensor(1, channels_edge))
            self.init_biases()
        if self.use_normalized_incidence:
            self.alpha = alpha
            self.beta = beta
            self.n_nodes, self.n_edges = self.incidence_1.shape
            self.compute_normalization_matrices()
            self.normalize_incidence_matrices()

    def compute_normalization_matrices(self) -> None:
        """Compute the normalization matrices for the incidence matrices."""
        B1 = self.incidence_1.to_dense()
        edge_cardinality = (B1.sum(0)) ** self.alpha
        node_cardinality = (B1.sum(1)) ** self.beta

        # Compute D0_left_alpha_inverse
        self.D0_left_alpha_inverse = torch.zeros(self.n_nodes, self.n_nodes)
        for i_node in range(self.n_nodes):
            self.D0_left_alpha_inverse[i_node, i_node] = 1 / (
                edge_cardinality[B1[i_node, :].bool()].sum()
            )

        # Compute D1_left_beta_inverse
        self.D1_left_beta_inverse = torch.zeros(self.n_edges, self.n_edges)
        for i_edge in range(self.n_edges):
            self.D1_left_beta_inverse[i_edge, i_edge] = 1 / (
                node_cardinality[B1[:, i_edge].bool()].sum()
            )

        # Compute D1_right_alpha
        self.D1_right_alpha = torch.diag(edge_cardinality)

        # Compute D0_right_beta
        self.D0_right_beta = torch.diag(node_cardinality)
        return

    def normalize_incidence_matrices(self) -> None:
        """Normalize the incidence matrices."""
        self.incidence_1 = (
            self.D0_left_alpha_inverse
            @ self.incidence_1.to_dense()
            @ self.D1_right_alpha
        ).to_sparse()
        self.incidence_1_transpose = (
            self.D1_left_beta_inverse
            @ self.incidence_1_transpose.to_dense()
            @ self.D0_right_beta
        ).to_sparse()
        return

    def init_biases(self) -> None:
        """Initialize the bias."""
        for bias in [self.bias_0_to_1, self.bias_1_to_0]:
            if self.bias_init == "xavier_uniform":
                torch.nn.init.xavier_uniform_(bias, gain=self.bias_gain)
            elif self.bias_init == "xavier_normal":
                torch.nn.init.xavier_normal_(bias, gain=self.bias_gain)

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.conv_1_to_0.reset_parameters()
        self.conv_0_to_1.reset_parameters()
        if self.use_bias:
            self.init_biases()

    def forward(self, x_0, x_1):
        r"""Forward computation.

        The forward pass was initially proposed in [DSB20]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        The equations of one layer of this neural network are given by:
        .. math::
        \begin{align*}
        &🟥 $\quad m_{y \rightarrow x}^{(0 \rightarrow 1)} = \sigma((B_1^T \cdot W^{(0)})_{xy} \cdot h_y^{t,(0)} \cdot \Theta^{t,(0)} + b^{t,(0)})$

        &🟥 $\quad m_{y \rightarrow x}^{(1 \rightarrow 0)}  = \sigma((B_1 \cdot W^{(1)})_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1)} + b^{t,(1)})$

        &🟧 $\quad m_x^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(0 \rightarrow 1)}$

        &🟧 $\quad m_x^{(1 \rightarrow 0)}  = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(1 \rightarrow 0)}$

        &🟩 $\quad m_x^{(0)}  = m_x^{(1 \rightarrow 0)}$

        &🟩 $\quad m_x^{(1)}  = m_x^{(0 \rightarrow 1)}$

        &🟦 $\quad h_x^{t+1,(0)}  = m_x^{(0)}$

        &🟦 $\quad h_x^{t+1,(1)} = m_x^{(1)}$
        \end{align*}

        References
        ----------
        .. [DSB20] Dong, Sawin, Bengio.
            HNHN: Hypergraph networks with hyperedge neurons.
            Graph Representation Learning and Beyond Workshop at ICML 2020
            https://grlplus.github.io/papers/40.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, channels_node]
            Input features on the hypernodes
        x_1 : torch.Tensor, shape=[n_edges, channels_edge]
            Input features on the hyperedges

        Returns
        -------
        x_0 : torch.Tensor, shape=[n_nodes, channels_node]
            Output features on the hypernodes
        x_1 : torch.Tensor, shape=[n_edges, channels_edge]
            Output features on the hyperedges
        """
        # Move incidence matrices to device
        self.incidence_1 = self.incidence_1.to(x_0.device)
        self.incidence_1_transpose = self.incidence_1_transpose.to(x_0.device)
        # Compute output hyperedge features
        x_1_tp1 = self.conv_0_to_1(x_0, self.incidence_1_transpose)  # nodes to edges
        if self.use_bias:
            x_1_tp1 += self.bias_0_to_1
        # Compute output hypernode features
        x_0_tp1 = self.conv_1_to_0(x_1, self.incidence_1)  # edges to nodes
        if self.use_bias:
            x_0_tp1 += self.bias_1_to_0
        return torch.sigmoid(x_0_tp1), torch.sigmoid(x_1_tp1)
