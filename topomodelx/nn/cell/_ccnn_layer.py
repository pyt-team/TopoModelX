import torch
from torch.nn import Linear
from torch import Tensor

from topomodelx.base.conv import Conv
from topomodelx.base.aggregation import Aggregation

class CCNNLayer(torch.nn.Module): 

    r"TODO: my description"

    def __init__(self, 
                in_channels: int,
                out_channels: int,
                harmonic: bool = False,
                aggr_func: str = "sum",
                update_func: str = "relu",
                **kwargs):

        super().__init__()
        
        # Filtering branches
        self.irrotational = Conv(
            in_channels=in_channels, out_channels=out_channels
        )
        self.solenoidal = Conv(
            in_channels=in_channels, out_channels=out_channels
        )
        if harmonic:
            # TODO: temp version
            self.harmonic = Linear(in_channels, out_channels, bias=False)
        else:
            self.register_parameter('harmonic', None)

        # between-neighborhood aggregation and update
        self.aggregation = Aggregation(aggr_func=aggr_func, update_func=update_func)

        self.reset_parameters()

    def reset_parameters(self):
        self.irrotational.reset_parameters()
        self.solenoidal.reset_parameters()
        if hasattr(self, "harmonic") and self.harmonic is not None:
            self.harmonic.reset_parameters()

    def forward(self, x, lower_neighborhood, upper_neighborhood) -> Tensor:

        r"TODO: my description"

        # message and within-neighborhood aggregation
        irrotational_x = self.irrotational(x, lower_neighborhood)
        solenoidal_x = self.solenoidal(x, upper_neighborhood)

        neighborhoods = [irrotational_x, solenoidal_x]

        if self.harmonic is not None:
            harmonic_x = self.harmonic(x)
            neighborhoods.append(harmonic_x)

        # between-neighborhood aggregation and update
        out = self.aggregation(neighborhoods)

        return out
    
if __name__ == "__main__":

    # dimensional test
    ccnn = CCNNLayer(3, 4, harmonic=True)
    x = torch.randn(10, 3)
    lower_neighborhood = torch.randn(10, 10).to_sparse()
    upper_neighborhood = torch.randn(10, 10).to_sparse()
    out = ccnn(x, lower_neighborhood, upper_neighborhood)
    print(out.shape)
        