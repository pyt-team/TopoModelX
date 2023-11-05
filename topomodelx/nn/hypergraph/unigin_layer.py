"""Implementation of UniGIN layer from Huang et. al.: UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks."""
import torch


class UniGINLayer(torch.nn.Module):
    """Layer of UniGIN.

    Implementation of UniGIN layer proposed in [1]_.


    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    eps : float
        Constant in GIN Update equation.
    train_eps : bool
        Whether to make eps a trainable parameter.

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
        https://arxiv.org/abs/2304.10031
    """

    def __init__(
        self,
        in_channels,
        eps: float = 0.,
        train_eps: bool = False,
    ) -> None:
        super().__init__()
        
        self.initial_eps = eps
        if train_eps:
            self.eps = torch.nn.Parameter(torch.Tensor([eps]))
        else:
            self.register_buffer("eps", torch.Tensor([eps]))
        
        self.linear = torch.nn.Linear(in_channels, in_channels)

    def forward(self, x_0, incidence_1):
        r"""[1]_ initially proposed the forward pass.

        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of three steps.

        1. Every hyper-edge sums up the features of its constituent edges:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)}  = B_1^T \cdot h_y^{t, (0)}\\
            &🟧 \quad m_z^{(0 \rightarrow 1)}  = \sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\\
            \end{align*}

        2. The message to the nodes is the sum of the messages from the incident hyper-edges.

        .. math::
            \begin{align*}
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1\rightarrow0)}  = \sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)}\\
            \end{align*}

        3. The node features are then updated using the GIN update equation:

        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_{x}^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = \Theta^t \cdot ((1+\eps)\cdot h_x^{t,(0)}+m_x^{(0)})
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
        incidence_1_transpose = incidence_1.to_dense().T.to_sparse()
        # First pass fills in features of edges by adding features of constituent nodes
        x_1 = torch.sparse.mm(incidence_1_transpose.float(), x_0)
        # Second pass fills in features of nodes by adding features of the incident edges
        m_1_0 = torch.sparse.mm(incidence_1.float(), x_1)
        # Update node features using GIN update equation
        x_0 = self.linear((1 + self.eps) * x_0 + m_1_0)
        return (x_0, x_1)
