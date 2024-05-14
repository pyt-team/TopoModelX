"""Cell Attention Network layer."""

from collections.abc import Callable
from typing import Literal

import torch
from torch import nn, topk
from torch.nn import Linear, Parameter, init
from torch.nn import functional as F

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter_add, scatter_sum


def softmax(src, index, num_cells: int):
    r"""Compute the softmax of the attention coefficients.

    Parameters
    ----------
    src : torch.Tensor, shape = (n_k_cells, heads)
        Attention coefficients.
    index : torch.Tensor, shape = (n_k_cells)
        Indices of the target nodes.
    num_cells : int
        Number of cells in the batch.

    Returns
    -------
    torch.Tensor, shape = (n_k_cells, heads)
        Softmax of the attention coefficients.

    Notes
    -----
    There should be of a default implementation of softmax in the utils file.
    Subtracting the maximum element in it from all elements to avoid overflow
    and underflow.
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

    Parameters
    ----------
    neighborhood : torch.sparse_coo_tensor, shape = (n_k_cells, n_k_cells)
        Neighborhood matrix.

    Returns
    -------
    torch.sparse_coo_tensor, shape = (n_k_cells, n_k_cells)
        Neighborhood matrix with self-loops.

    Notes
    -----
    Add to utils file.
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
    the Cell Attention Network (CAN) [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Number of input channels of the node signal.
    heads : int
        Number of attention heads.
    signal_lift_activation : Callable
        Activation function applied to the lifted signal.
    signal_lift_dropout : float
        Dropout rate applied to the lifted signal.

    References
    ----------
    .. [1] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks (2022).
        Paper: https://arxiv.org/pdf/2209.08179.pdf
        Repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int,
        signal_lift_activation: Callable,
        signal_lift_dropout: float,
    ) -> None:
        super().__init__()

        self.in_channels_0 = in_channels_0
        self.att_parameter = nn.Parameter(torch.empty(size=(2 * in_channels_0, heads)))
        self.signal_lift_activation = signal_lift_activation
        self.signal_lift_dropout = signal_lift_dropout

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = nn.init.calculate_gain("relu")
        nn.init.xavier_uniform_(self.att_parameter.data, gain=gain)

    def message(self, x_source, x_target=None):
        """Construct a message from source 0-cells to target 1-cell.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (num_edges, in_channels_0)
            Node signal of the source 0-cells.
        x_target : torch.Tensor, shape = (num_edges, in_channels_0)
            Node signal of the target 1-cell.

        Returns
        -------
        torch.Tensor, shape = (num_edges, heads)
            Edge signal.
        """
        # Concatenate source and target node feature vectors
        node_features_stacked = torch.cat((x_source, x_target), dim=1)

        # Compute the output edge signal by applying the activation function
        edge_signal = torch.einsum(
            "ij,jh->ih", node_features_stacked, self.att_parameter
        )  # (num_edges, heads)
        return self.signal_lift_activation(edge_signal)

    def forward(self, x_0, adjacency_0) -> torch.Tensor:  # type: ignore[override]
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels_0)
            Node signal.
        adjacency_0 : torch.Tensor, shape = (num_nodes, num_nodes)
            Sparse neighborhood matrix.

        Returns
        -------
        torch.Tensor, shape = (num_edges, 1)
            Edge signal.
        """
        # Extract source and target nodes from the graph's edge index
        source, target = adjacency_0.indices()  # (num_edges,)

        # Extract the node signal of the source and target nodes
        x_source = x_0[source]  # (num_edges, in_channels_0)
        x_target = x_0[target]  # (num_edges, in_channels_0)

        # Compute the edge signal
        return self.message(x_source, x_target)  # (num_edges, 1)


class MultiHeadLiftLayer(nn.Module):
    r"""Multi Head Attentional Lift Layer.

    Multi Head Attentional Lift Layer adapted from the official implementation of the Cell Attention Network (CAN) [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Number of input channels.
    heads : int, optional
        Number of attention heads.
    signal_lift_activation : Callable, optional
        Activation function to apply to the output edge signal.
    signal_lift_dropout : float, optional
        Dropout rate to apply to the output edge signal.
    signal_lift_readout : str, optional
        Readout method to apply to the output edge signal.
    """

    def __init__(
        self,
        in_channels_0: int,
        heads: int = 1,
        signal_lift_activation: Callable = torch.relu,
        signal_lift_dropout: float = 0.0,
        signal_lift_readout: str = "cat",
    ) -> None:
        super().__init__()

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

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        self.lifts.reset_parameters()

    def forward(self, x_0, adjacency_0, x_1=None) -> torch.Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (num_nodes, in_channels_0)
            Node signal.
        adjacency_0 : torch.Tensor, shape = (2, num_edges)
            Edge index.
        x_1 : torch.Tensor, shape = (num_edges, in_channels_1), optional
            Edge signal.

        Returns
        -------
        torch.Tensor, shape = (num_edges, heads + in_channels_1)
            Lifted node signal.

        Notes
        -----
        .. math::
            \begin{align*}
            &🟥 \quad m_{(y,z) \rightarrow x}^{(0 \rightarrow 1)}
                = \alpha(h_y, h_z) = \Theta(h_z||h_y)\\
            &🟦 \quad h_x^{(1)}
                = \phi(h_x, m_x^{(1)})
            \end{align*}
        """
        # Lift the node signal for each attention head
        attention_heads_x_1 = self.lifts(x_0, adjacency_0)

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
    r"""Attentional Pooling Layer.

    Attentional Pooling Layer adapted from the official implementation of the Cell Attention Network (CAN) [1]_.

    Parameters
    ----------
    k_pool : float in (0, 1]
        The pooling ratio i.e, the fraction of r-cells to keep after the pooling operation.
    in_channels_0 : int
        Number of input channels of the input signal.
    signal_pool_activation : Callable
        Activation function applied to the pooled signal.
    readout : bool, optional
        Whether to apply a readout operation to the pooled signal.
    """

    def __init__(
        self,
        k_pool: float,
        in_channels_0: int,
        signal_pool_activation: Callable,
        readout: bool = True,
    ) -> None:
        super().__init__()

        self.k_pool = k_pool
        self.in_channels_0 = in_channels_0
        self.readout = readout
        # Learnable attention parameter for the pooling operation
        self.att_pool = nn.Parameter(torch.empty(size=(in_channels_0, 1)))
        self.signal_pool_activation = signal_pool_activation

        # Initialize the attention parameter using Xavier initialization
        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reinitialize learnable parameters using Xavier uniform initialization."""
        gain = init.calculate_gain("relu")
        init.xavier_uniform_(self.att_pool.data, gain=gain)

    def forward(  # type: ignore[override]
        self, x, down_laplacian_1, up_laplacian_1
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_r_cells, in_channels_r)
            Input r-cell signal.
        down_laplacian_1 : torch.Tensor
            Lower neighborhood matrix.
        up_laplacian_1 : torch.Tensor
            Upper neighbourhood matrix.

        Returns
        -------
        torch.Tensor
            Pooled r_cell signal of shape (n_r_cells, in_channels_r).

        Notes
        -----
        .. math::
            \begin{align*}
            &🟥 \quad m_{x}^{(r)}
                = \gamma^t(h_x^t) = \tau^t (a^t\cdot h_x^t)\\
            &🟦 \quad h_x^{t+1,(r)}
                = \phi^t(h_x^t, m_{x}^{(r)}), \forall x\in \mathcal C_r^{t+1}
            \end{align*}
        """
        # Compute the output r-cell signal by applying the activation function
        Zp = torch.einsum("nc,ce->ne", x, self.att_pool)
        # Apply top-k pooling to the r-cell signal
        _, top_indices = topk(Zp.view(-1), int(self.k_pool * Zp.size(0)))
        # Rescale the pooled signal
        Zp = self.signal_pool_activation(Zp)
        out = x[top_indices] * Zp[top_indices]

        # Readout operation
        if self.readout:
            out = scatter_add(out, top_indices, dim=0, dim_size=x.size(0))[top_indices]

        # Update lower and upper neighborhood matrices with the top-k pooled r-cells
        down_laplacian_1_modified = torch.index_select(down_laplacian_1, 0, top_indices)
        down_laplacian_1_modified = torch.index_select(
            down_laplacian_1_modified, 1, top_indices
        )
        up_laplacian_1_modified = torch.index_select(up_laplacian_1, 0, top_indices)
        up_laplacian_1_modified = torch.index_select(
            up_laplacian_1_modified, 1, top_indices
        )
        # return sparse matrices of neighborhood
        return (
            out,
            down_laplacian_1_modified.to_sparse().float().coalesce(),
            up_laplacian_1_modified.to_sparse().float().coalesce(),
        )


class MultiHeadCellAttention(MessagePassing):
    """Attentional Message Passing v1.

    Attentional Message Passing from Cell Attention Network (CAN) [1]_ following the attention mechanism proposed in GAT [2]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : float
        Dropout rate applied to the output signal.
    heads : int
        Number of attention heads.
    concat : bool
        Whether to concatenate the output of each attention head.
    att_activation : Callable
        Activation function to use for the attention weights.
    add_self_loops : bool, optional
        Whether to add self-loops to the adjacency matrix.
    aggr_func : Literal["sum", "mean", "add"], default="sum"
        Aggregation function to use.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method for the weights of the layer.

    Notes
    -----
    If there are no non-zero values in the neighborhood, then the neighborhood is empty and forward returns zeros Tensor.

    References
    ----------
    .. [2] Veličković, Cucurull, Casanova, Romero, Liò and Bengio.
        Graph attention networks (2017).
        https://arxiv.org/pdf/1710.10903.pdf
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
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
    ) -> None:
        super().__init__(aggr_func=aggr_func, att=True, initialization=initialization)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.att_activation = att_activation
        self.heads = heads
        self.concat = concat
        self.dropout = dropout
        self.add_self_loops = add_self_loops

        self.lin = torch.nn.Linear(in_channels, out_channels, bias=False)
        self.att_weight_src = Parameter(torch.Tensor(1, heads, out_channels // heads))
        self.att_weight_dst = Parameter(torch.Tensor(1, heads, out_channels // heads))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the layer parameters."""
        torch.nn.init.xavier_uniform_(self.att_weight_src)
        torch.nn.init.xavier_uniform_(self.att_weight_dst)
        self.lin.reset_parameters()

    def message(self, x_source):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_k_cells, channels)
            Input features on the r-cell of the cell complex.

        Returns
        -------
        torch.Tensor, shape = (n_k_cells, heads, in_channels)
            Messages on source cells.
        """
        # Compute the linear transformation on the source features
        x_message = self.lin(x_source).view(
            -1, self.heads, self.out_channels // self.heads
        )  # (n_k_cells, H, C)

        # compute the source and target messages
        x_source_per_message = x_message[self.source_index_j]  # (|n_k_cells|, H, C)
        x_target_per_message = x_message[self.target_index_i]  # (|n_k_cells|, H, C)
        # compute the attention coefficients
        alpha = self.attention(
            x_source_per_message, x_target_per_message
        )  # (|n_k_cells|, H)

        # for each head, Aggregate the messages
        return x_source_per_message * alpha[:, :, None]  # (|n_k_cells|, H, C)

    def attention(self, x_source, x_target):
        """Compute attention weights for messages.

        Parameters
        ----------
        x_source : torch.Tensor, shape = [n_k_cells, in_channels]
            Source node features.
        x_target : torch.Tensor, shape = [n_k_cells, in_channels]
            Target node features.

        Returns
        -------
        torch.Tensor, shape = [n_k_cells, heads]
            Attention weights.
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
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def forward(self, x_source, neighborhood):
        """Forward pass.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_k_cells, channels)
            Input features on the r-cell of the cell complex.
        neighborhood : torch.sparse, shape = (n_k_cells, n_k_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_k).

        Returns
        -------
        torch.Tensor, shape = (n_k_cells, channels)
            Output features on the r-cell of the cell complex.
        """
        # If there are no non-zero values in the neighborhood, then the neighborhood is empty. -> return zero tensor
        if not neighborhood.values().nonzero().size(0) > 0 and self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels),
                device=x_source.device,
            )  # (n_k_cells, H * C)
        if not neighborhood.values().nonzero().size(0) > 0 and not self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels // self.heads),
                device=x_source.device,
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
            return aggregated_message.view(-1, self.out_channels)  # (n_k_cells, H * C)

        return aggregated_message.mean(dim=1)  # (n_k_cells, C)


class MultiHeadCellAttention_v2(MessagePassing):
    """Attentional Message Passing v2.

    Attentional Message Passing from Cell Attention Network (CAN) [1]_ following the attention mechanism proposed in GATv2 [3]_

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    dropout : float
        Dropout rate applied to the output signal.
    heads : int
        Number of attention heads.
    concat : bool
        Whether to concatenate the output of each attention head.
    att_activation : Callable
        Activation function to use for the attention weights.
    add_self_loops : bool, optional
        Whether to add self-loops to the adjacency matrix.
    aggr_func : Literal["sum", "mean", "add"], default="sum"
        Aggregation function to use.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method for the weights of the layer.
    share_weights : bool, optional
        Whether to share the weights between the attention heads.

    Notes
    -----
    If there are no non-zero values in the neighborhood, then the neighborhood is empty.

    References
    ----------
    .. [3] Brody, Alon, Yahav.
        How attentive are graph attention networks? (2022).
        https://arxiv.org/pdf/2105.14491.pdf
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        dropout: float,
        heads: int,
        concat: bool,
        att_activation: torch.nn.Module,
        add_self_loops: bool = True,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        share_weights: bool = False,
    ) -> None:
        super().__init__(
            aggr_func=aggr_func,
            att=True,
            initialization=initialization,
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
                in_channels, out_channels, bias=False
            )
        else:
            self.lin_src = torch.nn.Linear(in_channels, out_channels, bias=False)
            self.lin_dst = torch.nn.Linear(in_channels, out_channels, bias=False)

        self.att_weight = Parameter(torch.Tensor(1, heads, out_channels // heads))

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the layer parameters."""
        torch.nn.init.xavier_uniform_(self.att_weight)
        self.lin_src.reset_parameters()
        self.lin_dst.reset_parameters()

    def message(self, x_source):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_k_cells, channels)
            Input features on the r-cell of the cell complex.

        Returns
        -------
        Tensor, shape = (n_k_cells, heads, in_channels)
            Messages on source cells.
        """
        # Compute the linear transformation on the source features
        x_src_message = self.lin_src(x_source).view(
            -1, self.heads, self.out_channels // self.heads
        )  # (n_k_cells, H, C)

        # Compute the linear transformation on the source features
        x_dst_message = self.lin_dst(x_source).view(
            -1, self.heads, self.out_channels // self.heads
        )  # (n_k_cells, H, C)

        # Get the source and target projections of the neighborhood
        x_source_per_message = x_src_message[self.source_index_j]  # (|n_k_cells|, H, C)
        x_target_per_message = x_dst_message[self.target_index_i]  # (|n_k_cells|, H, C)

        # concatenate the source and target projections of the neighborhood
        x_message = x_source_per_message + x_target_per_message  # (|n_k_cells|, H, C)

        # Compute the attention coefficients
        alpha = self.attention(x_message)  # (|n_k_cells|, H)

        # for each head, Aggregate the messages
        return x_source_per_message * alpha[:, :, None]  # (|n_k_cells|, H, C)

    def attention(self, x_source):
        """Compute attention weights for messages.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (|n_k_cells|, heads, in_channels)
            Source node features.

        Returns
        -------
        torch.Tensor, shape = (n_k_cells, heads)
            Attention weights.
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
        return F.dropout(alpha, p=self.dropout, training=self.training)

    def forward(self, x_source, neighborhood):
        """Forward pass.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_k_cells, channels)
            Input features on the r-cell of the cell complex.
        neighborhood : torch.sparse, shape = (n_k_cells, n_k_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_k), [up, down].

        Returns
        -------
        torch.Tensor, shape = (n_k_cells, channels)
            Output features on the r-cell of the cell complex.
        """
        # If there are no non-zero values in the neighborhood, then the neighborhood is empty. -> return zero tensor
        if not neighborhood.values().nonzero().size(0) > 0 and self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels),
                device=x_source.device,
            )  # (n_k_cells, H * C)
        if not neighborhood.values().nonzero().size(0) > 0 and not self.concat:
            return torch.zeros(
                (x_source.shape[0], self.out_channels // self.heads),
                device=x_source.device,
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
            return aggregated_message.view(-1, self.out_channels)  # (n_k_cells, H * C)

        return aggregated_message.mean(dim=1)  # (n_k_cells, C)


class CANLayer(torch.nn.Module):
    r"""Layer of the Cell Attention Network (CAN) model.

    The CAN layer considers an attention convolutional message passing though the upper and lower neighborhoods of the cell.
    Additionally, a skip connection can be added to the output of the layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features on n-cells.
    out_channels : int
        Dimension of output.
    heads : int, default=1
        Number of attention heads.
    dropout : float, optional
        Dropout probability of the normalized attention coefficients.
    concat : bool, default=True
        If True, the output of each head is concatenated. Otherwise, the output of each head is averaged.
    skip_connection : bool, default=True
        If True, skip connection is added.
    att_activation : Callable, default=torch.nn.LeakyReLU()
        Activation function applied to the attention coefficients.
    add_self_loops : bool, optional
        If True, self-loops are added to the neighborhood matrix.
    aggr_func : Literal["mean", "sum"], default="sum"
        Between-neighborhood aggregation function applied to the messages.
    update_func : Literal["relu", "sigmoid", "tanh", None], default="relu"
        Update function applied to the messages.
    version : Literal["v1", "v2"], default="v1"
        Version of the layer, by default "v1" which is the same as the original CAN layer. While "v2" has the same attetion mechanism as the GATv2 layer.
    share_weights : bool, default=False
        This option is valid only for "v2". If True, the weights of the linear transformation applied to the source and target features are shared, by default False.
    **kwargs : optional
        Additional arguments of CAN layer.

    Notes
    -----
    Add_self_loops is preferred to be False. If necessary, the self-loops should be added to the neighborhood matrix in the preprocessing step.
    """

    lower_att: MultiHeadCellAttention | MultiHeadCellAttention_v2
    upper_att: MultiHeadCellAttention | MultiHeadCellAttention_v2

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        heads: int = 1,
        dropout: float = 0.0,
        concat: bool = True,
        skip_connection: bool = True,
        att_activation: torch.nn.Module | None = None,
        add_self_loops: bool = True,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "relu",
        version: Literal["v1", "v2"] = "v1",
        share_weights: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()

        if att_activation is None:
            att_activation = torch.nn.LeakyReLU()

        assert in_channels > 0, ValueError("Number of input channels must be > 0")
        assert out_channels > 0, ValueError("Number of output channels must be > 0")
        assert heads > 0, ValueError("Number of heads must be > 0")
        assert out_channels % heads == 0, ValueError(
            "Number of output channels must be divisible by the number of heads"
        )
        assert dropout >= 0.0 and dropout <= 1.0, ValueError("Dropout must be in [0,1]")

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
            out_channels = out_channels if concat else out_channels // heads
            self.lin = Linear(in_channels, out_channels, bias=False)
            self.eps = 1 + 1e-6

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self) -> None:
        """Reset the parameters of the layer."""
        self.lower_att.reset_parameters()
        self.upper_att.reset_parameters()
        if hasattr(self, "lin"):
            self.lin.reset_parameters()

    def forward(self, x, down_laplacian_1, up_laplacian_1) -> torch.Tensor:
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_k_cells, channels)
            Input features on the r-cell of the cell complex.
        down_laplacian_1 : torch.sparse, shape = (n_k_cells, n_k_cells)
            Lower neighborhood matrix mapping r-cells to r-cells (A_k_low).
        up_laplacian_1 : torch.sparse, shape = (n_k_cells, n_k_cells)
            Upper neighborhood matrix mapping r-cells to r-cells (A_k_up).

        Returns
        -------
        torch.Tensor, shape = (n_k_cells, out_channels)
            Output features on the r-cell of the cell complex.

        Notes
        -----
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
        """
        # message and within-neighborhood aggregation
        lower_x = self.lower_att(x, down_laplacian_1)
        upper_x = self.upper_att(x, up_laplacian_1)

        # skip connection
        if hasattr(self, "lin"):
            w_x = self.lin(x) * self.eps

        # between-neighborhood aggregation and update
        return (
            self.aggregation([lower_x, upper_x, w_x])
            if hasattr(self, "lin")
            else self.aggregation([lower_x, upper_x])
        )
