import torch

from topomodelx.nn.base import _MessagePassing


class MessagePassingConv(_MessagePassing):
    """Message passing: steps 1 and 2.
    Everything that is intra neighborhood.
    We will have one of this per neighborhood.
    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    intra_agg : string
        Aggregation method.
        (Inter-neighborhood).
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_agg="sum",
        initialization="xavier_uniform",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            intra_agg=intra_agg,
        )
        self.initialization = initialization

    def forward(self, x, neighborhood):
        weighted_h = torch.mm(x, self.weight)
        message = torch.spmm(neighborhood, weighted_h)
        if self.intra_agg == "sum":
            return message
        neighborhood_size = torch.sum(neighborhood, axis=1)
        return torch.einsum("i,ij->ij", 1 / neighborhood_size, message)
