"""Implementation of UniSAGE layer from Huang et. al.: UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks."""
import torch
from torch import nn
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class UniSAGELayer(torch.nn.Module):
    """Layer of UniSAGE.

    Implementation of UniSAGE layer proposed in [JJ21]_.

    References
    ----------
    ..  [JJ21]Jing Huang and Jie Yang. UniGNN: a unified framework for graph and hypergraph neural networks.
        In Proceedings of the Thirtieth International Joint Conference on Artificial Intelligence, IJCAI-21,
        2021.
        https://arxiv.org/pdf/2105.00956.pdf

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    e_aggr : string
        Aggregator function for hyperedges. Defaults to "sum", other options are
        "sum", "mean", "amax", or "amin".
    v_aggr : string
        Aggregator function for nodes. Defaults to " mean", other options are
        "sum", "mean", "amax", or "amin".
    use_bn : boolean
        Whether to use bathnorm after the linear transformation.
    """

    def _validate_aggr(self, aggr):
        if aggr not in {"sum", "mean", "amax", "amin"}:
            raise ValueError(
                f"Unsupported aggregator: {aggr}, should be 'sum', 'mean', 'amax', or 'amin'"
            )

    def __init__(
        self,
        in_channels,
        out_channels,
        e_aggr="sum",
        v_aggr="mean",
        use_bn=False,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.bn = nn.BatchNorm1d(out_channels) if use_bn else None
        self.linear = nn.Linear(in_channels, out_channels)
        self.v_aggr = v_aggr
        self.e_aggr = e_aggr

        self._validate_aggr(v_aggr)
        self._validate_aggr(e_aggr)

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.linear.reset_parameters()
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

        2. The message to the nodes is the sum of the messages from the incident hyper-edges.
        .. math::
            \begin{align*}
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1\rightarrow0)}  = \operatorname{AGGREGATE}_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)}\\
            \end{align*}

        3. The node features are then updated using the SAGE update equation:
        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_{x}^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = (h_x^{t,(0)} + m_x^{(0)})
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
        x_0 = self.linear(x_0)
        if self.bn is not None:
            x_0 = self.bn(x_0)

        # Use sparse CSR to enable reduce operation
        if incidence_1.layout != torch.sparse_csr:
            incidence_1_transpose = incidence_1.T.to_sparse_csr()
            incidence_1 = incidence_1.to_sparse_csr()
        else:
            # Transpose generates CSC tensor, thus we have to convert to csr
            incidence_1_transpose = incidence_1.transpose(1, 0).to_sparse_csr()
        # First pass fills in features of edges by adding features of constituent nodes
        m_0_1 = torch.sparse.mm(incidence_1_transpose.float(), x_0, reduce=self.e_aggr)
        # Second pass fills in features of nodes by adding features of the incident edges
        m_1_0 = torch.sparse.mm(incidence_1.float(), m_0_1, reduce=self.v_aggr)
        return x_0 + m_1_0
