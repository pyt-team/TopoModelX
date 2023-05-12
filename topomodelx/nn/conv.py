import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import _MessagePassing


class MessagePassingConv(_MessagePassing):
    """Message passing: steps 1, 2, and 3.
    Everything that is intra neighborhood.
    We will have one of this per neighborhood.
    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    intra_aggr: string
        Aggregation method.
        (Inter-neighborhood).
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        neighborhood,
        initialization="xavier_uniform",
        update="sigmoid",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        self.neighborhood = neighborhood
        self.initialization = initialization
        self.update = update

        self.weight = Parameter(torch.Tensor(in_channels, out_channels))
        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        if self.initialization == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.initialization == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=gain)

        elif self.initialization == "uniform":
            stdv = 1.0 / torch.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)

        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.initialization}' is not supported"
            )

        return self.weight

    def forward(self, x):
        weighted_x = torch.mm(x, self.weight)
        print("weighted_x", weighted_x.shape)
        print("neighborhood", self.neighborhood.shape)
        message = torch.mm(self.neighborhood, weighted_x)
        if self.intra_aggr == "sum":
            return message
        neighborhood_size = torch.sum(self.neighborhood, axis=1)
        if self.update == "sigmoid":
            return F.sigmoid(torch.einsum("i,ij->ij", 1 / neighborhood_size))
        return torch.einsum("i,ij->ij", 1 / neighborhood_size, message)
