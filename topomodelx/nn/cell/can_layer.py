"""Cell Attention Network layer."""

import torch
from torch import Tensor
from torch.nn import Linear, Parameter
from torch.nn import functional as F

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import MessagePassing
from topomodelx.utils.scatter import scatter_sum


class MultiHeadCellAttention(MessagePassing):
    """Attentional Message Passing from Cell Attention Network (CAN) [CAN22]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    heads : int
        Number of attention heads.
    concat : bool
        Whether to concatenate the output of each attention head.
    att_activation : string
        Activation function to use for the attention weights.
    aggr_func : string
        Aggregation function to use. Options are "sum", "mean", "max".
    initialization : string
        Initialization method for the weights of the layer.

    Notes
    -----
    [] If there are no non-zero values in the neighborhood, then the neighborhood is empty.

    References
    ----------
    [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
        paper: https://arxiv.org/pdf/2209.08179.pdf
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        dropout,
        heads,
        concat,
        att_activation,
        aggr_func="sum",
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
            aggr_func=aggr_func,
        )

        assert att_activation in [
            "leaky_relu",
            "elu",
            "tanh",
        ], "Invalid activation function."

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_activation = att_activation
        self.heads = heads
        self.concat = concat
        self.dropout = dropout

        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_weight_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_weight_dst = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer parameters."""
        torch.nn.init.xavier_uniform_(self.att_weight_src)
        torch.nn.init.xavier_uniform_(self.att_weight_dst)
        self.lin.reset_parameters()

    def attention(self, x_source):
        """Compute attention weights for messages.

        Parameters
        ----------
        x_source : torch.Tensor
            Source node features. Shape: [n_k_cells, in_channels]

        Returns
        -------
        alpha : torch.Tensor
            Attention weights. Shape: [n_k_cells, heads]
        """
        alpha_src = (self.x_source_per_message * self.att_weight_src).sum(
            dim=-1
        )  # (E, H)
        alpha_dst = (self.x_target_per_message * self.att_weight_dst).sum(
            dim=-1
        )  # (E, H)

        alpha = alpha_src + alpha_dst  # (E, H)

        # Apply activation function # TODO: add more activation functions? Pass directly the function to avoid if-if?
        if self.att_activation == "elu":
            alpha = torch.nn.functional.elu(alpha)
        if self.att_activation == "leaky_relu":
            alpha = torch.nn.functional.leaky_relu(alpha)  # TODO: add negative slope?
        if self.att_activation == "tanh":
            alpha = torch.nn.functional.tanh(alpha)

        # Normalize the attention coefficients
        self.softmax(alpha, self.target_index_i, x_source.shape[0])
        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha

    def softmax(self, src: torch.Tensor, index: torch.Tensor, num_cells: int):
        """Compute the softmax of the attention coefficients."""
        # TODO: in the utils there's no scatter_max
        # The scatter_max function is used to make the softmax function numerically stable
        # by subtracting the maximum value in each row before computing the exponential.
        # src_max = scatter_max(src, index, dim=0, dim_size=num_nodes)[0][index]
        src = torch.exp(src)  # src - src_max
        src_sum = scatter_sum(src, index, dim=0, dim_size=num_cells)[index]
        return src / (src_sum + 1e-16)

    def forward(self, x_source, neighborhood):
        """Forward pass.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_k_cells, channels]
            Input features on the k-cell of the cell complex.
        neighborhood : torch.sparse, shape=[n_k_cells, n_k_cells]
            Neighborhood matrix mapping k-cells to k-cells (A_k). [up, down]

        Returns
        -------
        _ : torch.Tensor, shape=[n_k_cells, channels]
            Output features on the k-cell of the cell complex.
        """
        # If there are no non-zero values in the neighborhood, then the neighborhood is empty. -> return zero tensor
        if not neighborhood.values().nonzero().size(0) > 0 and self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels * self.heads),
                device=x_source.device,
            )  # (E, H * C)
        if not neighborhood.values().nonzero().size(0) > 0 and not self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels), device=x_source.device
            )  # (E, C)

        # Compute the linear transformation on the source features
        x_message = self.lin(x_source).view(-1, self.heads, self.out_channels)

        # Compute the attention coefficients
        (
            self.target_index_i,
            self.source_index_j,
        ) = neighborhood.indices()  # (E, 1), (E, 1)
        # TODO: we have to consider also the weights of the "edges" (?) -> neighborhood.values()
        self.x_source_per_message = x_message[self.source_index_j]  # (E, H, C)
        self.x_target_per_message = x_message[self.target_index_i]  # (E, H, C)
        alpha = self.attention(x_message)  # (E, H)

        # for each head, Aggregate the messages # TODO: check if the aggregation is correct
        message = self.x_source_per_message * alpha[:, :, None]  # (E, H, C)
        aggregated_message = self.aggregate(message)  # (E, H, C)

        # if concat true, concatenate the messages for each head. Otherwise, average the messages for each head.
        if self.concat:
            return aggregated_message.view(
                -1, self.heads * self.out_channels
            )  # (E, H * C)

        return aggregated_message.mean(dim=1)  # (E, C)


class CANLayer(torch.nn.Module):
    r"""Layer of the Cell Attention Network (CAN) model.

    The CAN layer considers an attention convolutional message passing though the upper and lower neighborhoods of the cell.
    Additionally a skip connection can be added to the output of the layer.

    ..  math::
        \mathcal N_k \in  \mathcal N = \{A_{\uparrow, r}, A_{\downarrow, r}\}

    ..  math::
        \begin{align*}
        &🟥 \quad m_{y \rightarrow x}^{(r \rightarrow r)} = M_{\mathcal N_k}(h_x^{t}, h_y^{t}, \Theta^{t}_k)\\
        &🟧 \quad m_x^{(r \rightarrow r)} = \text{AGG}_{y \in \mathcal{N}_k(x)}(m_{y \rightarrow x}^{(r \rightarrow r)})\\
        &🟩 \quad m_x^{(r)} = \text{AGG}_{\mathcal{N}_k\in\mathcal N}m_x^{(r \rightarrow r)}\\
        &🟦 \quad h_x^{t+1,(r)} = U^{t}(h_x^{t}, m_x^{(r)})
        \end{align*}

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks.
        (2022) paper: https://arxiv.org/pdf/2209.08179.pdf

    Parameters
    ----------
    in_channels : int
        Dimension of input features on n-cells.
    out_channels : int
        Dimension of output
    heads : int, optional
        Number of attention heads, by default 1
    dropout : float, optional
        Dropout probability, by default 0.0
    concat : bool, optional
        If True, the output of each head is concatenated. Otherwise, the output of each head is averaged, by default True
    skip_connection : bool, optional
        If True, skip connection is added, by default True
    att_activation : str, optional
        Activation function for the attention coefficients, by default "leaky_relu". ["elu", "leaky_relu", "tanh"]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        skip_connection: bool = True,
        att_activation: str = "leaky_relu",
        aggr_func="sum",
        update_func: str = "relu",
        **kwargs,
    ):
        super().__init__()

        # TODO: add all the assertions
        assert in_channels > 0, ValueError("Number of input channels must be > 0")
        assert out_channels > 0, ValueError("Number of output channels must be > 0")

        # lower attention
        self.lower_att = MultiHeadCellAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            heads=heads,
            att_activation=att_activation,
            concat=concat,
        )

        # upper attention
        self.upper_att = MultiHeadCellAttention(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            heads=heads,
            att_activation=att_activation,
            concat=concat,
        )

        # linear transformation
        if skip_connection:
            out_channels = out_channels * heads if concat else out_channels
            self.lin = Linear(in_channels, out_channels, bias=False)
            self.eps = 1 + 1e-6

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the parameters of the layer."""
        self.lower_att.reset_parameters()
        self.upper_att.reset_parameters()
        if hasattr(self, "lin"):
            self.lin.reset_parameters()

    def forward(self, x, lower_neighborhood, upper_neighborhood) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape=[n_k_cells, channels]
            Input features on the k-cell of the cell complex.
        lower_neighborhood : torch.sparse
            shape=[n_k_cells, n_k_cells]
            Lower neighborhood matrix mapping k-cells to k-cells (A_k_low).
        upper_neighborhood : torch.sparse
            shape=[n_k_cells, n_k_cells]
            Upper neighborhood matrix mapping k-cells to k-cells (A_k_up).

        Returns
        -------
        _ : torch.Tensor, shape=[n_k_cells, out_channels]
        """
        # message and within-neighborhood aggregation
        lower_x = self.lower_att(x, lower_neighborhood)
        upper_x = self.upper_att(x, upper_neighborhood)

        # skip connection
        if hasattr(self, "lin"):
            w_x = self.lin(x) * self.eps

        # between-neighborhood aggregation and update
        out = (
            self.aggregation([lower_x, upper_x, w_x])
            if hasattr(self, "lin")
            else self.aggregation([lower_x, upper_x])
        )

        return out
