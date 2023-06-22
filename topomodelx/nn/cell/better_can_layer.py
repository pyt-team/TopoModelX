import torch
from torch.nn import Linear
from torch import Tensor

from topomodelx.base.conv import Conv
from topomodelx.base.aggregation import Aggregation

class CANLayer(torch.nn.Module): 

    r"TODO: my description"

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                aggr_func: str = "sum",
                update_func: str = "relu",
                **kwargs):

        super().__init__()
        
        # Filtering branches

        # irrotational branch
        self.irrotational = Conv(
            in_channels=in_channels, out_channels=out_channels, att=True
        )

        # solenoidal branch
        self.solenoidal = Conv(
            in_channels=in_channels, out_channels=out_channels, att=True
        )

        # harmonic branch
        self.harmonic = Linear(in_channels, out_channels, bias=False)
        self.eps = 1 + 1e-6

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self):
        self.irrotational.reset_parameters()
        self.solenoidal.reset_parameters()
        self.harmonic.reset_parameters()

    def forward(self, x, lower_neighborhood, upper_neighborhood) -> Tensor:

        r"TODO: my description"

        # message and within-neighborhood aggregation
        irrotational_x = self.irrotational(x, lower_neighborhood)
        solenoidal_x = self.solenoidal(x, upper_neighborhood)
        harmonic_x = self.harmonic(x)*self.eps

        # between-neighborhood aggregation and update
        out = self.aggregation([irrotational_x, solenoidal_x, harmonic_x])

        return out
    
if __name__ == "__main__":

    # dimensional test
    can = CANLayer(3, 4)
    x = torch.randn(10, 3)
    lower_neighborhood = torch.randn(10, 10).to_sparse()
    upper_neighborhood = torch.randn(10, 10).to_sparse()
    out = can(x, lower_neighborhood, upper_neighborhood)
    print(out.shape)