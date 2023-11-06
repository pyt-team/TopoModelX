"""Implementation of UniSAGE layer from Huang et. al.: UniGNN: a Unified Framework for Graph and Hypergraph Neural Networks."""

from typing import Literal

import torch
from topomodelx.base.conv import Conv

class UniSAGELayer(torch.nn.Module):
    """Layer of UniSAGE proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    hidden_channels : int
        Dimension of output features.
    e_aggr : Literal["sum", "mean",], default="sum"
        Aggregator function for hyperedges.
    v_aggr : Literal["sum", "mean",], default="mean"
        Aggregator function for nodes.
    use_bn : boolean
        Whether to use bathnorm after the linear transformation.

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

    def _validate_aggr(self, aggr):
        if aggr not in {"sum", "mean",}:
            raise ValueError(
                f"Unsupported aggregator: {aggr}, should be 'sum', 'mean',"
            )

    def __init__(
        self,
        in_channels,
        hidden_channels,
        e_aggr: Literal["sum", "mean",] = "sum",
        v_aggr: Literal["sum", "mean",] = "mean",
        use_bn: bool = False,
    ) -> None:
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.bn = torch.nn.BatchNorm1d(hidden_channels) if use_bn else None
        
        # Initial linear transformation
        self.linear = torch.nn.Linear(in_channels, hidden_channels)
        
        # Define the aggregation
        self.v_aggr = v_aggr
        self.e_aggr = e_aggr

        self._validate_aggr(v_aggr)
        self._validate_aggr(e_aggr)

        self.vertex2edge = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr_norm=False if self.e_aggr=="sum" else True,
            with_linear_transform=False, 
            )

        self.edge2vertex = Conv(
            in_channels=hidden_channels,
            out_channels=hidden_channels,
            aggr_norm=False if self.v_aggr=="sum" else True,
            with_linear_transform=False,
            )

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.linear.reset_parameters()
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
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)}  = B_1 \cdot m_z^{(0 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{(1\rightarrow0)}  = \operatorname{AGGREGATE}_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1\rightarrow0)}\\
            \end{align*}

        Third, the node features are then updated using the SAGE update equation:

        .. math::
            \begin{align*}
            &🟩 \quad m_x^{(0)}  = m_{x}^{(1\rightarrow0)}\\
            &🟦 \quad h_x^{t+1,(0)}  = (h_x^{t,(0)} + m_x^{(0)})
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
        x_0 = self.linear(x_0)
        if self.bn is not None:
            x_0 = self.bn(x_0)

        x_1 =  self.vertex2edge(x_0, incidence_1.transpose(1, 0))
        m_1_0 = self.edge2vertex(x_1, incidence_1)
        x_0 = x_0 + m_1_0
        
        return (x_0, x_1)
