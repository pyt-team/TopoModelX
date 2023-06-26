import torch
from torch.nn import Linear, Parameter
from torch import Tensor

from topomodelx.base.conv import MessagePassing
from topomodelx.base.aggregation import Aggregation


class CANMessagePassing(MessagePassing):
    r"""Attentional Message Passing from Cell Attention Network (CAN). [CAN22]_

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
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
        att_activation,
        aggr_func="sum",
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
            aggr_func=aggr_func,
        )

        assert att_activation in ["leaky_relu", "elu", "tanh"]

        self.in_channels = in_channels
        self.out_channels = out_channels

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.att_weight = Parameter(
            torch.Tensor(
                2 * self.out_channels,
            )
        )

        self.reset_parameters()

    def attention(self, x_source, att_activation="leaky_relu"):
        """Compute attention weights for messages. [CAN22]_

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape=[n_target_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.Tensor, shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = x_source[self.target_index_i]

        # Concatenate source and target features
        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        # Compute attention weights
        x_att = torch.matmul(x_source_target_per_message, self.att_weight)

        # Apply activation function
        if att_activation == "elu":
            return torch.nn.functional.elu(x_att)
        elif att_activation == "leaky_relu":
            return torch.nn.functional.leaky_relu(x_att)
        elif att_activation == "tanh":
            return torch.nn.functional.tanh(x_att)
        else:
            raise NotImplementedError(
                f"Activation function {att_activation} not implemented."
            )

    def forward(self, x_source, neighborhood):
        r"""Forward pass.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_k_cells, channels]
            Input features on the k-cell of the cell complex.
        neighborhood : torch.sparse, shape=[n_k_cells, n_k_cells]
            Neighborhood matrix mapping k-cells to k-cells (A_k). [up, down]

        Returns
        -------
        out : torch.Tensor, shape=[n_k_cells, channels]
        """

        # Compute the linear transformation on the source features
        x_message = torch.mm(x_source, self.weight)

        # Compute the attention coefficients
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        attention_values = self.attention(x_message)
        neighborhood = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention_values * neighborhood.values(),
            size=neighborhood.shape,
        )

        # Normalize the attention coefficients
        neighborhood = torch.sparse.softmax(neighborhood, dim=1)

        # Aggregate the messages
        # If there are no non-zero values in the neighborhood, then the neighborhood is empty
        neighborhood_values = neighborhood.values()
        if (
            neighborhood_values.nonzero().size(0) > 0
        ):  # Check if there are any non-zero values
            x_message = x_message.index_select(-2, self.source_index_j)
            x_message = neighborhood_values.view(-1, 1) * x_message
            out = self.aggregate(x_message)
        else:  # Special case for all zero neighborhood_values
            # Create a tensor of the correct shape filled with zeros
            out = torch.zeros(
                (x_message.shape[0], x_message.shape[1]), device=x_message.device
            )

        return out


class CANMultiHeadMessagePassing(torch.nn.Module):
    """
    Multi-head version of the CANMessagePassing class.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        num_heads,
        att_activation,
        concat=True,
        **kwargs,
    ):
        super(CANMultiHeadMessagePassing, self).__init__()

        self.heads = torch.nn.ModuleList()
        for _ in range(num_heads):
            self.heads.append(
                CANMessagePassing(in_channels, out_channels, att_activation)
            )

        self.concat = concat

    def reset_parameters(self):
        for head in self.heads:
            head.reset_parameters()

    def forward(self, x_source, neighborhood):
        out = []
        for head in self.heads:
            out.append(head(x_source, neighborhood))

        if self.concat:
            # Concatenate the output along the feature dimension
            return torch.cat(out, dim=1)
        else:
            # Take the mean of the output
            return torch.mean(torch.stack(out), dim=0)


class CANLayer(torch.nn.Module):
    r"""Layer of the Cell Attention Network (CAN) model.

    The CAN layer considers an attention convolutional message passing though the upper and lower neighborhoods of the cell.

    ..  math::
        \mathcal N_k \in  \mathcal N = \{A_{\uparrow, r}, A_{\downarrow, r}\}

    ..  math::
        \begin{align*}            
        &🟥 \quad m_{y \rightarrow x}^{(r \rightarrow r)} = M_{\mathcal N_k}(h_x^{t}, h_y^{t}, \Theta^{t}_k)\\ 
        &🟧 \quad m_x^{(r \rightarrow r)} = \text{AGG}_{y \in \mathcal{N}_k(x)}(m_{y \rightarrow x}^{(r \rightarrow r)})\\
        &🟩 \quad m_x^{(r)} = \text{AGG}_{\mathcal{N}_k\in\mathcal N}m_x^{(r \rightarrow r)}\\            
        &🟦 \quad h_x^{t+1,(r)} = U^{t}(h_x^{t}, m_x^{(r)})
        \end{align*}

    Notes
    -----
    [] Add multi-head attention

    References
    ----------
    [CAN22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).
        paper: https://arxiv.org/pdf/2209.08179.pdf

    Parameters
    ----------
    in_channels : int
        Dimension of input features on n-cells.
    out_channels : int
        Dimension of output
    skip_connection : bool, optional
        If True, skip connection is added, by default True
    att_activation : str, optional
        Activation function for the attention coefficients, by default "leaky_relu". ["elu", "leaky_relu", "tanh"]
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        num_heads: int = 1,
        concat: bool = True,
        skip_connection: bool = True,
        att_activation: str = "leaky_relu",
        aggr_func="sum",
        update_func: str = "relu",
        **kwargs,
    ):
        super().__init__()

        assert in_channels > 0, ValueError("Number of input channels must be > 0")
        assert out_channels > 0, ValueError("Number of output channels must be > 0")

        # lower attention
        self.lower_att = CANMultiHeadMessagePassing(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            att_activation=att_activation,
            concat=concat,
        )

        # upper attention
        self.upper_att = CANMultiHeadMessagePassing(
            in_channels=in_channels,
            out_channels=out_channels,
            num_heads=num_heads,
            att_activation=att_activation,
            concat=concat,
        )

        # linear transformation
        if skip_connection:
            out_channels = out_channels * num_heads if concat else out_channels
            self.lin = Linear(in_channels, out_channels, bias=False)
            self.eps = 1 + 1e-6

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reset the parameters of the layer."""
        self.lower_att.reset_parameters()
        self.upper_att.reset_parameters()
        if hasattr(self, "lin"):
            self.lin.reset_parameters()

    def forward(self, x, lower_neighborhood, upper_neighborhood) -> Tensor:
        r"""Forward pass.

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
