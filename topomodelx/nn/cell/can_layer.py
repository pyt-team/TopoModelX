import torch
from torch.nn import Linear, Parameter
from torch import Tensor

from topomodelx.base.conv import MessagePassing
from topomodelx.base.aggregation import Aggregation


class CANConv(MessagePassing):
    r"""Cell Attention Network (CAN) convolutional layer.

    Parameters
    ----------
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        update_func=None,

        initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.att_weight = Parameter(
            torch.Tensor(
                2 * self.out_channels,
            )
        )

        self.reset_parameters()

    def forward(self, x_source, neighborhood):
        r"""Forward pass.

        Parameters
        ----------

        Returns
        -------
        """
        x_message = torch.mm(x_source, self.weight)

        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        attention_values = self.attention(x_message)
        neighborhood = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention_values * neighborhood.values(),
            size=neighborhood.shape,
        )
            
        neighborhood = torch.sparse.softmax(neighborhood, dim=1).to_dense()

        x_message = torch.mm(neighborhood, x_message)

        return x_message

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

    References
    ----------
    [GBTDLSB22] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa. “Cell attention networks”. In: arXiv preprint arXiv:2209.08179 (2022).

    Parameters
    ----------
    in_channels : int
        Dimension of input features on n-cells.
    out_channels : int
        Dimension of output
    aggr_func : str = "sum"
        The aggregation function between-neighborhoods. ["sum", "mean"]
    update_func : str = "relu"
        The update function within-neighborhoods. ["relu", "sigmoid"]
    """

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                aggr_func: str = "sum",
                update_func: str = "relu",
                **kwargs):

        super().__init__()

        # lower attention
        self.lower_att = CANConv(
            in_channels=in_channels, out_channels=out_channels
        )

        # upper attention
        self.upper_att = CANConv(
            in_channels=in_channels, out_channels=out_channels
        )

        # linear transformation
        self.lin = Linear(in_channels, out_channels, bias=False)
        self.eps = 1 + 1e-6

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self):
        self.lower_att.reset_parameters()
        self.upper_att.reset_parameters()
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

        w_x = self.lin(x)*self.eps

        # between-neighborhood aggregation and update
        out = self.aggregation([lower_x, upper_x, w_x])

        return out