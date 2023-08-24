"""Cell Attention Network layer."""

from typing import Callable

import torch
from torch import Tensor, nn, topk
from torch.nn import Linear, Parameter
from torch.nn import functional as F
from torch.nn import init

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter_add, scatter_sum


def softmax(src, index, num_cells):
    r"""Compute the softmax of the attention coefficients.

    Notes
    -----
    There should be of a default implementation of softmax in the utils file.
    Subtracting the maximum element in it from all elements to avoid overflow
    and underflow.

    Parameters
    ----------
    src : torch.Tensor
        Attention coefficients. Shape: [n_k_cells, heads]
    index : torch.Tensor
        Indices of the target nodes. Shape: [n_k_cells]
    num_cells : int
        Number of cells in the batch.

    Returns
    -------
    _ : torch.Tensor
        Softmax of the attention coefficients. Shape: [n_k_cells, heads]
    """
    src_max = src.max(dim=0, keepdim=True)[0]  # (1, H)
    src -= src_max  # (|n_k_cells|, H)
    src_exp = torch.exp(src)  # (|n_k_cells|, H)
    src_sum = scatter_sum(src_exp, index, dim=0, dim_size=num_cells)[
        index
    ]  # (|n_k_cells|, H)
    return src_exp / (src_sum + 1e-16)  # (|n_k_cells|, H)


def add_self_loops(neighborhood):
    """Add self-loops to the neighborhood matrix.

    Notes
    -----
    Add to utils file.

    Parameters
    ----------
    neighborhood : torch.sparse_coo_tensor
        Neighborhood matrix. Shape: [n_k_cells, n_k_cells]

    Returns
    -------
    _ : torch.sparse_coo_tensor
        Neighborhood matrix with self-loops. Shape: [n_k_cells, n_k_cells]
    """
    N = neighborhood.shape[0]
    cell_index, cell_weight = neighborhood._indices(), neighborhood._values()
    # create loop index
    loop_index = torch.arange(0, N, dtype=torch.long, device=neighborhood.device)
    loop_index = loop_index.unsqueeze(0).repeat(2, 1)
    # add loop index to neighborhood
    cell_index = torch.cat([cell_index, loop_index], dim=1)
    cell_weight = torch.cat(
        [cell_weight, torch.ones(N, dtype=torch.float, device=neighborhood.device)]
    )

    return torch.sparse_coo_tensor(
        indices=cell_index,
        values=cell_weight,
        size=(N, N),
    ).coalesce()


class LiftLayer(MessagePassing):
    """Attentional Lift Layer.

    This is adapted from the official implementation of
    the CeLL Attention Network (CAN) [CAN22]_.

    Parameters
    ----------
    in_channels_0: int
        Number of input channels of the node signal.
    heads: int
        Number of attention heads.
    signal_lift_activation: Callable
        Activation function applied to the lifted signal.
    signal_lift_dropout: float
        Dropout rate applied to the lifted signal.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int,
        signal_lift_activation: Callable,
        signal_lift_dropout: float,
    ):
        super(LiftLayer, self).__init__()

        self.in_channels_0 = in_channels_0
        self.att = nn.Parameter(torch.empty(size=(2 * in_channels_0, heads)))
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.att.data, gain=gain)

    def message(self, x_source, x_target=None):
        """Construct message from source 0-cells to target 1-cell."""
        # Concatenate source and target node feature vectors
        node_features_stacked = torch.cat(
            (x_source, x_target), dim=1
        )  # (num_edges, 2 * in_channels_0)

        # Compute the output edge signal by applying the activation function
        edge_signal = torch.einsum(
            "ij,jh->ih", node_features_stacked, self.att
        )  # (num_edges, heads)
        edge_signal = self.signal_lift_activation(edge_signal)  # (num_edges, heads)

        return edge_signal  # (num_edges, heads)

    def forward(self, x_0, neighborhood_0_to_0) -> Tensor:
        """Forward pass.

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0)
        neighborhood_0_to_0: torch.Tensor
            Sparse neighborhood matrix of shape (num_nodes, num_nodes)

        Returns
        -------
        _: torch.Tensor
            Edge signal of shape (num_edges, 1)
        """
        # Extract source and target nodes from the graph's edge index
        source, target = neighborhood_0_to_0.indices()  # (num_edges,)

        # Extract the node signal of the source and target nodes
        x_source = x_0[source]  # (num_edges, in_channels_0)
        x_target = x_0[target]  # (num_edges, in_channels_0)

        # Compute the edge signal
        return self.message(x_source, x_target)  # (num_edges, 1)


class MultiHeadLiftLayer(nn.Module):
    r"""Multi Head Attentional Lift Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can

    Parameters
    ----------
    in_channels_0: int
        Number of input channels.
    heads: int, optional
        Number of attention heads.
    signal_lift_activation: Callable, optional
        Activation function to apply to the output edge signal.
    signal_lift_dropout: float, optional
        Dropout rate to apply to the output edge signal.
    signal_lift_readout: str, optional
        Readout method to apply to the output edge signal.
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int = 1,
        signal_lift_activation: Callable = torch.relu,
        signal_lift_dropout: float = 0.0,
        signal_lift_readout: str = "cat",
        *args,
        **kwargs,
    ):
        super(MultiHeadLiftLayer, self).__init__()

        assert heads > 0, ValueError("Number of heads must be > 0")
        assert signal_lift_readout in [
            "cat",
            "sum",
            "avg",
            "max",
        ], "Invalid readout method."

        self.in_channels_0 = in_channels_0
        self.heads = heads
        self.signal_lift_readout = signal_lift_readout
        self.signal_lift_dropout = signal_lift_dropout
        self.signal_lift_activation = signal_lift_activation
        self.lifts = LiftLayer(
            in_channels_0=in_channels_0,
            heads=heads,
            signal_lift_activation=signal_lift_activation,
            signal_lift_dropout=signal_lift_dropout,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        self.lifts.reset_parameters()

    def forward(self, x_0, neighborhood_0_to_0, x_1=None) -> torch.Tensor:
        r"""Forward pass.

        .. math::
            \begin{align*}
            &🟥 \quad m_{(y,z) \rightarrow x}^{(0 \rightarrow 1)}
                = \alpha(h_y, h_z) = \Theta(h_z||h_y)\\
            &🟦 \quad h_x^{(1)}
                = \phi(h_x, m_x^{(1)})
            \end{align*}

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0)
        neighborhood_0_to_0: torch.Tensor
            Edge index of shape (2, num_edges)
        x_1: torch.Tensor, optional
            Node signal of shape (num_edges, in_channels_1)

        Returns
        -------
        _: torch.Tensor
            Lifted node signal of shape (num_edges, heads + in_channels_1)
        """
        # Lift the node signal for each attention head
        attention_heads_x_1 = self.lifts(x_0, neighborhood_0_to_0)

        # Combine the output edge signals using the specified readout strategy
        readout_methods = {
            "cat": lambda x: x,
            "sum": lambda x: x.sum(dim=1)[:, None],
            "avg": lambda x: x.mean(dim=1)[:, None],
            "max": lambda x: x.max(dim=1).values[:, None],
        }
        combined_x_1 = readout_methods[self.signal_lift_readout](attention_heads_x_1)

        # Apply dropout to the combined edge signal
        combined_x_1 = F.dropout(
            combined_x_1, self.signal_lift_dropout, training=self.training
        )

        # Concatenate the lifted node signal with the original node signal if is not None
        if x_1 is not None:
            combined_x_1 = torch.cat(
                (combined_x_1, x_1), dim=1
            )  # (num_edges, heads + in_channels_1)

        return combined_x_1


class PoolLayer(MessagePassing):
    r"""Attentional Pooling Layer adapted from the official implementation of the CeLL Attention Network (CAN) [CAN22]_.

    Parameters
    ----------
    k_pool: float
        The pooling ratio i.e, the fraction of edges to keep after the pooling operation. (0,1]
    in_channels_0: int
        Number of input channels of the input signal.
    signal_pool_activation: Callable
        Activation function applied to the pooled signal.
    readout: bool, optional
        Whether to apply a readout operation to the pooled signal.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks. (2022)
        paper: https://arxiv.org/pdf/2209.08179.pdf
        repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        k_pool: float,
        in_channels_0: int,
        signal_pool_activation: Callable,
        readout: bool = True,
    ):
        super(PoolLayer, self).__init__()

        self.k_pool = k_pool
        self.in_channels_0 = in_channels_0
        self.readout = readout
        # Learnable attention parameter for the pooling operation
        self.att_pool = nn.Parameter(torch.empty(size=(in_channels_0, 1)))
        self.signal_pool_activation = signal_pool_activation

        # Initialize the attention parameter using Xavier initialization
        self.reset_parameters()

    def reset_parameters(self):
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = init.calculate_gain("relu")
        init.xavier_uniform_(self.att_pool.data, gain=gain)

    def forward(self, x_0, lower_neighborhood, upper_neighborhood) -> Tensor:
        r"""Forward pass.

        .. math::
            \begin{align*}
            &🟥 \quad m_{x}^{(r)}
                = \gamma^t(h_x^t) = \tau^t (a^t\cdot h_x^t)\\
            &🟦 \quad h_x^{t+1,(r)}
                = \phi^t(h_x^t, m_{x}^{(r)}), \forall x\in \mathcal C_r^{t+1}
            \end{align*}

        Parameters
        ----------
        x_0: torch.Tensor
            Node signal of shape (num_nodes, in_channels_0).
        neighborhood_0_to_0: torch.Tensor
            Neighborhood matrix of shape (num_edges, 2).

        Returns
        -------
        out: torch.Tensor
            Pooled node signal of shape (num_pooled_nodes, in_channels_0).
        """
        # Compute the output edge signal by applying the activation function
        Zp = torch.einsum("nc,ce->ne", x_0, self.att_pool)
        # Apply top-k pooling to the edge signal
        _, top_indices = topk(Zp.view(-1), int(self.k_pool * Zp.size(0)))
        # Rescale the pooled signal
        Zp = self.signal_pool_activation(Zp)
        out = x_0[top_indices] * Zp[top_indices]

        # Readout operation
        if self.readout:
            out = scatter_add(out, top_indices, dim=0, dim_size=x_0.size(0))[
                top_indices
            ]

        # Update lower and upper neighborhood matrices with the top-k pooled edges
        lower_neighborhood_modified = torch.index_select(
            lower_neighborhood, 0, top_indices
        )
        lower_neighborhood_modified = torch.index_select(
            lower_neighborhood_modified, 1, top_indices
        )
        upper_neighborhood_modified = torch.index_select(
            upper_neighborhood, 0, top_indices
        )
        upper_neighborhood_modified = torch.index_select(
            upper_neighborhood_modified, 1, top_indices
        )
        # return sparse matrices of neighborhood
        return (
            out,
            lower_neighborhood_modified.to_sparse().float().coalesce(),
            upper_neighborhood_modified.to_sparse().float().coalesce(),
        )


class MultiHeadCellAttention(MessagePassing):
    """Attentional Message Passing from Cell Attention Network (CAN) [CAN22] following the attention mechanism proposed in GAT [GAT17].

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
    att_activation : Callable
        Activation function to use for the attention weights.
    add_self_loops : bool, optional
        Whether to add self-loops to the adjacency matrix.
    aggr_func : string, optional
        Aggregation function to use. Options are "sum", "mean", "max".
    initialization : string, optional
        Initialization method for the weights of the layer.

    Notes
    -----
    [] If there are no non-zero values in the neighborhood, then the neighborhood is empty and forward returns zeros Tensor.

    References
    ----------
    [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
        paper: https://arxiv.org/pdf/2209.08179.pdf

    [GAT17] Veličković, Cucurull, Casanova, Romero, Liò and Bengio. “Graph attention networks”. In: arXiv preprint arXiv:1710.10903 (2017).
        paper: https://arxiv.org/pdf/1710.10903.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        heads: int,
        concat: bool,
        att_activation: torch.nn.Module,
        add_self_loops: bool = False,
        aggr_func: str = "sum",
        initialization: str = "xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
            aggr_func=aggr_func,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_activation = att_activation
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
        self.att_weight_src = Parameter(torch.Tensor(1, heads, out_channels))
        self.att_weight_dst = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer parameters."""
        torch.nn.init.xavier_uniform_(self.att_weight_src)
        torch.nn.init.xavier_uniform_(self.att_weight_dst)
        self.lin.reset_parameters()

    def message(self, x_source):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_k_cells, channels]
            Input features on the k-cell of the cell complex.

        Returns
        -------
        _ : Tensor, shape=[n_k_cells, heads, in_channels]
            Messages on source cells.
        """
        # Compute the linear transformation on the source features
        x_message = self.lin(x_source).view(
            -1, self.heads, self.out_channels
        )  # (n_k_cells, H, C)

        # compute the source and target messages
        x_source_per_message = x_message[self.source_index_j]  # (|n_k_cells|, H, C)
        x_target_per_message = x_message[self.target_index_i]  # (|n_k_cells|, H, C)
        # compute the attention coefficients
        alpha = self.attention(
            x_source_per_message, x_target_per_message
        )  # (|n_k_cells|, H)

        # for each head, Aggregate the messages
        message = x_source_per_message * alpha[:, :, None]  # (|n_k_cells|, H, C)
        return message

    def attention(self, x_source, x_target):
        """Compute attention weights for messages.

        Parameters
        ----------
        x_source : torch.Tensor
            Source node features. Shape: [n_k_cells, in_channels]
        x_target : torch.Tensor
            Target node features. Shape: [n_k_cells, in_channels]

        Returns
        -------
        _ : torch.Tensor
            Attention weights. Shape: [n_k_cells, heads]
        """
        # Compute attention coefficients
        alpha_src = torch.einsum(
            "ijk,tjk->ij", x_source, self.att_weight_src
        )  # (|n_k_cells|, H)
        alpha_dst = torch.einsum(
            "ijk,tjk->ij", x_target, self.att_weight_dst
        )  # (|n_k_cells|, H)

        alpha = alpha_src + alpha_dst

        # Apply activation function
        alpha = self.att_activation(alpha)

        # Normalize the attention coefficients
        alpha = softmax(alpha, self.target_index_i, x_source.shape[0])

        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha  # (|n_k_cells|, H)

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
            )  # (n_k_cells, H * C)
        if not neighborhood.values().nonzero().size(0) > 0 and not self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels), device=x_source.device
            )  # (n_k_cells, C)

        # Add self-loops to the neighborhood matrix if necessary
        if self.add_self_loops:
            neighborhood = add_self_loops(neighborhood)

        # returns the indices of the non-zero values in the neighborhood matrix
        (
            self.target_index_i,
            self.source_index_j,
        ) = neighborhood.indices()  # (|n_k_cells|, 1), (|n_k_cells|, 1)

        # compute message passing step
        message = self.message(x_source)  # (|n_k_cells|, H, C)
        # compute within-neighborhood aggregation step
        aggregated_message = self.aggregate(message)  # (n_k_cells, H, C)

        # if concat true, concatenate the messages for each head. Otherwise, average the messages for each head.
        if self.concat:
            return aggregated_message.view(
                -1, self.heads * self.out_channels
            )  # (n_k_cells, H * C)

        return aggregated_message.mean(dim=1)  # (n_k_cells, C)


class MultiHeadCellAttention_v2(MessagePassing):
    """Attentional Message Passing from Cell Attention Network (CAN) [CAN22] following the attention mechanism proposed in GATv2 [GATv2_22].

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
    att_activation : Callable
        Activation function to use for the attention weights.
    add_self_loops : bool, optional
        Whether to add self-loops to the adjacency matrix.
    aggr_func : string, optional
        Aggregation function to use. Options are "sum", "mean", "max".
    initialization : string, optional
        Initialization method for the weights of the layer.
    share_weights : bool, optional
        Whether to share the weights between the attention heads.

    Notes
    -----
    [] If there are no non-zero values in the neighborhood, then the neighborhood is empty.

    References
    ----------
    [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
        paper: https://arxiv.org/pdf/2209.08179.pdf

    [GATv2_22] Brody, Alon, Yahav. "How Attentive are Graph Attention Networks?" In: arXiv preprint arXiv:2105.14491 (2022).
        paper: https://arxiv.org/pdf/2105.14491.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        heads: int,
        concat: bool,
        att_activation: torch.nn.Module,
        add_self_loops: bool = False,
        aggr_func: str = "sum",
        initialization: str = "xavier_uniform",
        share_weights: bool = False,
    ):
        super().__init__(
            att=True,
            initialization=initialization,
            aggr_func=aggr_func,
        )

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_activation = att_activation
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        if share_weights:
            self.lin_src = self.lin_dst = torch.nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
        else:
            self.lin_src = torch.nn.Linear(
                in_channels, heads * out_channels, bias=False
            )
            self.lin_dst = torch.nn.Linear(
                in_channels, heads * out_channels, bias=False
            )

        self.att_weight = Parameter(torch.Tensor(1, heads, out_channels))

        self.reset_parameters()

    def reset_parameters(self):
        """Reset the layer parameters."""
        torch.nn.init.xavier_uniform_(self.att_weight)
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def message(self, x_source):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_k_cells, channels]
            Input features on the k-cell of the cell complex.

        Returns
        -------
        _ : Tensor, shape=[n_k_cells, heads, in_channels]
            Messages on source cells.
        """
        # Compute the linear transformation on the source features
        x_src_message = self.lin_src(x_source).view(
            -1, self.heads, self.out_channels
        )  # (n_k_cells, H, C)

        # Compute the linear transformation on the source features
        x_dst_message = self.lin_dst(x_source).view(
            -1, self.heads, self.out_channels
        )  # (n_k_cells, H, C)

        # Get the source and target projections of the neighborhood
        x_source_per_message = x_src_message[self.source_index_j]  # (|n_k_cells|, H, C)
        x_target_per_message = x_dst_message[self.target_index_i]  # (|n_k_cells|, H, C)

        # concatenate the source and target projections of the neighborhood
        x_message = x_source_per_message + x_target_per_message  # (|n_k_cells|, H, C)

        # Compute the attention coefficients
        alpha = self.attention(x_message)  # (|n_k_cells|, H)

        # for each head, Aggregate the messages
        message = x_source_per_message * alpha[:, :, None]  # (|n_k_cells|, H, C)

        return message

    def attention(self, x_source):
        """Compute attention weights for messages.

        Parameters
        ----------
        x_source : torch.Tensor
            Source node features. Shape: [|n_k_cells|, heads, in_channels]

        Returns
        -------
        alpha : torch.Tensor
            Attention weights. Shape: [n_k_cells, heads]
        """
        # Apply activation function
        x_source = self.att_activation(x_source)  # (|n_k_cells|, H, C)

        # Compute attention coefficients
        alpha = torch.einsum(
            "ijk,tjk->ij", x_source, self.att_weight
        )  # (|n_k_cells|, H)

        # Normalize the attention coefficients
        alpha = softmax(alpha, self.target_index_i, x_source.shape[0])

        # Apply dropout
        alpha = F.dropout(alpha, p=self.dropout, training=self.training)

        return alpha  # (|n_k_cells|, H)

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
            )  # (n_k_cells, H * C)
        if not neighborhood.values().nonzero().size(0) > 0 and not self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels), device=x_source.device
            )  # (n_k_cells, C)

        # Add self-loops to the neighborhood matrix if necessary
        if self.add_self_loops:
            neighborhood = add_self_loops(neighborhood)

        # Get the source and target indices of the neighborhood
        (
            self.target_index_i,
            self.source_index_j,
        ) = neighborhood.indices()  # (|n_k_cells|, 1), (|n_k_cells|, 1)

        # compute message passing step
        message = self.message(x_source)  # (|n_k_cells|, H, C)
        # compute within-neighborhood aggregation step
        aggregated_message = self.aggregate(message)  # (n_k_cells, H, C)

        # if concat true, concatenate the messages for each head. Otherwise, average the messages for each head.
        if self.concat:
            return aggregated_message.view(
                -1, self.heads * self.out_channels
            )  # (n_k_cells, H * C)

        return aggregated_message.mean(dim=1)  # (n_k_cells, C)


class CANLayer(torch.nn.Module):
    r"""Layer of the Cell Attention Network (CAN) model.

    The CAN layer considers an attention convolutional message passing though the upper and lower neighborhoods of the cell.
    Additionally a skip connection can be added to the output of the layer.

    References
    ----------
    .. [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks.
        (2022) paper: https://arxiv.org/pdf/2209.08179.pdf

    Notes
    -----
    Add_self_loops is preferred to be False. If necessary, the self-loops should be added to the neighborhood matrix in the preprocessing step.

    Parameters
    ----------
    in_channels : int
        Dimension of input features on n-cells.
    out_channels : int
        Dimension of output
    heads : int, optional
        Number of attention heads, by default 1
    dropout : float, optional
        Dropout probability of the normalized attention coefficients, by default 0.0
    concat : bool, optional
        If True, the output of each head is concatenated. Otherwise, the output of each head is averaged, by default True
    skip_connection : bool, optional
        If True, skip connection is added, by default True
    add_self_loops : bool, optional
        If True, self-loops are added to the neighborhood matrix, by default False
    att_activation : Callable, optional
        Activation function applied to the attention coefficients, by default torch.nn.LeakyReLU()
    aggr_func : str, optional
        Between-neighborhood aggregation function applied to the messages, by default "sum"
    update_func : str, optional
        Update function applied to the messages, by default "relu"
    version : str, optional
        Version of the layer, by default "v1" which is the same as the original CAN layer. While "v2" has the same attetion mechanism as the GATv2 layer.
    share_weights : bool, optional
        This option is valid only for "v2". If True, the weights of the linear transformation applied to the source and target features are shared, by default False
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        skip_connection: bool = True,
        att_activation: torch.nn.Module = torch.nn.LeakyReLU(),
        add_self_loops: bool = False,
        aggr_func="sum",
        update_func: str = "relu",
        version: str = "v1",
        share_weights: bool = False,
        **kwargs,
    ):
        super().__init__()

        assert in_channels > 0, ValueError("Number of input channels must be > 0")
        assert out_channels > 0, ValueError("Number of output channels must be > 0")
        assert heads > 0, ValueError("Number of heads must be > 0")
        assert dropout >= 0.0 and dropout <= 1.0, ValueError("Dropout must be in [0,1]")
        assert version in ["v1", "v2"], ValueError("Version must be 'v1' or 'v2'")
        # assert that shared weight is True only if version is v2
        assert share_weights is False or version == "v2", ValueError(
            "Shared weights is valid only for v2"
        )

        if version == "v1":
            # lower attention
            self.lower_att = MultiHeadCellAttention(
                in_channels=in_channels,
                out_channels=out_channels,
                add_self_loops=add_self_loops,
                dropout=dropout,
                heads=heads,
                att_activation=att_activation,
                concat=concat,
            )

            # upper attention
            self.upper_att = MultiHeadCellAttention(
                in_channels=in_channels,
                out_channels=out_channels,
                add_self_loops=add_self_loops,
                dropout=dropout,
                heads=heads,
                att_activation=att_activation,
                concat=concat,
            )

        elif version == "v2":
            # lower attention
            self.lower_att = MultiHeadCellAttention_v2(
                in_channels=in_channels,
                out_channels=out_channels,
                add_self_loops=add_self_loops,
                dropout=dropout,
                heads=heads,
                att_activation=att_activation,
                concat=concat,
                share_weights=share_weights,
            )

            # upper attention
            self.upper_att = MultiHeadCellAttention_v2(
                in_channels=in_channels,
                out_channels=out_channels,
                add_self_loops=add_self_loops,
                dropout=dropout,
                heads=heads,
                att_activation=att_activation,
                concat=concat,
                share_weights=share_weights,
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
        r"""Forward pass.

        .. math::
            \mathcal N = \{\mathcal N_1, \mathcal N_2\} = \{A_{\uparrow, r}, A_{\downarrow, r}\}

        .. math::
            \begin{align*}
            &🟥 \quad m_{(y \rightarrow x),k}^{(r)}
                = \alpha_k(h_x^t,h_y^t) = a_k(h_x^{t}, h_y^{t}) \cdot \psi_k^t(h_x^{t})\quad \forall \mathcal N_k\\
            &🟧 \quad m_{x,k}^{(r)}
                = \bigoplus_{y \in \mathcal{N}_k(x)}  m^{(r)}  _{(y \rightarrow x),k}\\
            &🟩 \quad m_{x}^{(r)}
                = \bigotimes_{\mathcal{N}_k\in\mathcal N}m_{x,k}^{(r)}\\
            &🟦 \quad h_x^{t+1,(r)}
                = \phi^{t}(h_x^t, m_{x}^{(r)})
            \end{align*}


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
