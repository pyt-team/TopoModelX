__all__ = ["Linear", "TensorLinear"]

from warnings import warn

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    r"""x -> xW +b"""

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        bias: bool = True,
        weight_initializer="xavier_uniform",
    ):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.init_scheme = weight_initializer

        self.weight = Parameter(torch.Tensor(out_ft, in_ft))  # flip the order
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain=1.414):
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self.init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=gain)

        elif self.init_scheme == "uniform":
            stdv = 1.0 / torch.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.init_scheme}' is not supported"
            )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The features.
        """
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_ft}, "
            f"{self.out_ft}, bias={self.bias is not None})"
        )


class TensorLinear(nn.Module):
    r"""x -> xW +b"""

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        in_ch_ft: int,
        out_ch_ft: int,
        bias: bool = True,
        weight_initializer="xavier_uniform",
    ):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.in_ch_ft = in_ch_ft
        self.out_ch_ft = out_ch_ft
        self.init_scheme = weight_initializer
        self.weight = Parameter(torch.Tensor(in_ft, out_ft, out_ch_ft))
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch_ft))

        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain=1.414):
        if self.bias is not None:
            nn.init.zeros_(self.bias)

        if self.init_scheme == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.init_scheme == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=gain)

        elif self.init_scheme == "uniform":
            stdv = 1.0 / torch.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.init_scheme}' is not supported"
            )

    def forward(self, x: Tensor) -> Tensor:
        r"""
        Args:
            x (Tensor): The features.
        """
        assert len(x.shape) == 3
        if x.shape[1] != self.in_ft:
            raise ValueError(
                "The feature dimension in the input tensor must "
                + f"equal to in_features in the model, got {x.shape[1]} and {self.in_ft}. "
            )

        if x.shape[-1] != self.in_ch_ft:
            raise ValueError(
                "The depth channel in the input tensor must be equal to the "
                + f"number of input channels in the model, got {x.shape[-1]} and {self.in_ch_ft} "
            )

        return torch.einsum("vid,iot->vot", x, self.weight) + self.bias


# X=torch.rand(10,4,20) #v_dim_depth

# F1=TensorLinear(4,12,20,30)

# t=F1(X)

# Z=torch.einsum('vid,iod->vo',x,W)
