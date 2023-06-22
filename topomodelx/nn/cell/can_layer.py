import torch
from torch.nn import Linear
from torch import Tensor

from topomodelx.base.conv import Conv
from topomodelx.base.aggregation import Aggregation

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
        self.lower_att = Conv(
            in_channels=in_channels, out_channels=out_channels, att=True
        )

        # upper attention
        self.upper_att = Conv(
            in_channels=in_channels, out_channels=out_channels, att=True
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

        r"TODO: my description"

        # message and within-neighborhood aggregation
        lower_x = self.lower_att(x, lower_neighborhood)
        upper_x = self.upper_att(x, upper_neighborhood)
        w_x = self.lin(x)*self.eps

        # between-neighborhood aggregation and update
        out = self.aggregation([lower_x, upper_x, w_x])

        return out
    
if __name__ == "__main__":

    # dimensional test
    can = CANLayer(3, 4)
    x = torch.randn(10, 3)
    lower_neighborhood = torch.randn(10, 10).to_sparse()
    upper_neighborhood = torch.randn(10, 10).to_sparse()
    out = can(x, lower_neighborhood, upper_neighborhood)
    print(out.shape)