import torch
from torch.nn import Linear
from torch import Tensor

from topomodelx.base.conv import Conv

class CCNNLayer(torch.nn.Module):
    r"TODO: my description"
    def __init__(self, 
                in_channels: int,
                out_channels: int,
                harmonic: bool = False,
                **kwargs):

        super().__init__()
        
        self.irrotational = Conv(
            in_channels=in_channels, out_channels=out_channels
        )
        self.solenoidal = Conv(
            in_channels=in_channels, out_channels=out_channels
        )

        if harmonic:
            # TODO: temp version
            self.harmonic = Linear(in_channels, out_channels, bias=False,
                                   weight_init="glorot_uniform")
        else:
            self.register_parameter('harmonic', None)

        self.reset_parameters()

    def reset_parameters(self):
        self.irrotational.reset_parameters()
        self.solenoidal.reset_parameters()
        if hasattr(self, "harmonic") and self.harmonic is not None:
            self.harmonic.reset_parameters()

    def forward(self, x, lower_neighborhood, upper_neighborhood) -> Tensor:

        r"TODO: my description"

        irrotational_x = self.irrotational(x, lower_neighborhood)
        solenoidal_x = self.solenoidal(x, upper_neighborhood)

        out = irrotational_x + solenoidal_x

        if self.harmonic is not None:
            out = out + self.harmonic(x)

        return out
    
if __name__ == "__main__":
    ccnn = CCNNLayer(3, 4)
    x = torch.randn(10, 3)
    lower_neighborhood = torch.randn(10, 10).to_sparse()
    upper_neighborhood = torch.randn(10, 10).to_sparse()
    out = ccnn(x, lower_neighborhood, upper_neighborhood)
    print(out.shape)
        