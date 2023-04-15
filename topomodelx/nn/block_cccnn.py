

from warnings import warn

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter

from topomodelx.nn.linear import TensorLinear


class BlockLTN(nn.Module):
    r"""
    Description
    -----------
        Args
        ----------
            input_ch: a list of positive integers,
                        each in_ch_i is the number of features in
                        the ith input dimension.
            target_ch :  number of features in the output dimension.
            n_channels: positive int, number of in depth channels.
            o_channels : positive int, number of out depth channels
            dropout: optional, default is 0.0.
            bias: optional, default is True.
            init_scheme: optional, default is xavier, other options : debug.

        Example
        -------
        >>>from stnets.util import coo_2_torch_tensor

        >>>simplices = [(0, 1, 2), (1, 2, 3), (2, 3), (1, 2, 4), (5, 3), (0, 4)]

        >>>HL = SimplicialComplex(simplices,mode="gudhi")

        >>>X=torch.rand(9,5,10)

        >>>B1=HL.get_boundary_operator(1)

        >>>B1=coo_2_torch_tensor(B1)

        >>>net=BlockLTN(5,10, 12,15)

        >>>Y=net(X,B1)"""

    def __init__(
        self,
        input_ch: int,
        in_depth_ch: int,
        target_ch: int,
        o_depth_ch: int,
        bias=True,
        init_scheme="xavier_uniform",
    ):
        super(BlockLTN, self).__init__()
        assert isinstance(input_ch, int)
        assert isinstance(target_ch, int)
        assert isinstance(in_depth_ch, int)
        assert isinstance(o_depth_ch, int)

        if input_ch < 1:
            raise ValueError(
                "Dimension of input features must be larger than or equal to 1."
            )
        if target_ch < 1:
            raise ValueError(
                "Dimension of output features must be larger than or equal to 1."
            )
        if in_depth_ch < 1:
            raise ValueError("Input of Channels must be larger than or equal to 1.")
        if o_depth_ch < 1:
            raise ValueError("Output of Channels must be larger than or equal to 1.")

        self.input_ch = input_ch
        self.in_depth_ch = in_depth_ch

        self.target_ch = target_ch
        self.o_depth_ch = o_depth_ch

        self.linear = nn.ModuleList(
            [
                TensorLinear(input_ch, target_ch, in_depth_ch, in_depth_ch)
                for i in range(0, o_depth_ch)
            ]
        )

    def forward(self, x: Tensor, G: Tensor) -> Tensor:

        if len(x.shape) == 2:
            x = x.unsqueeze(-1)

        if len(x.shape) != 3:
            raise ValueError(
                "input tensor must have at 2d or 3d tensor,"
                + f" however got input tensor of shape: {x.shape}."
            )
        if x.shape[0] != G.shape[-1]:
            raise ValueError(
                " number of in_simplices in the input tenor "
                + "must be equal to number of in_simplices "
                f"in the input operator, got: {x.shape[0]} and {G.shape[-1]}."
            )

        convolve_tensor = torch.stack([G for i in range(0, self.in_depth_ch)], dim=2)
        # convolve_tensor = convolve_tensor.to_dense()
        outputs = []
        for i in range(0, self.o_depth_ch):
            y = self.linear[i](x)
            output = torch.einsum("vcd,evd->ec", y, convolve_tensor)
            outputs.append(output)
        return torch.stack(outputs, dim=2)
