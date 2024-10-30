__all__ = ["Linear", "TensorLinear"]

import torch
import torch.nn.functional as F
from torch import Tensor, nn
from torch.nn.parameter import Parameter


class Linear(nn.Module):
    r"""Linear layer that applies the transformation: \( y = xW + b \)

    Parameters
    ----------
    in_ft : int
        Number of input features.
    out_ft : int
        Number of output features.
    bias : bool, optional
        If True, the layer will learn an additive bias. Default is True.
    weight_initializer : str, optional
        The initializer for the weights. Options are: 
        'xavier_uniform', 'xavier_normal', or 'uniform'. Default is 'xavier_uniform'.

    Examples
    --------
    >>> linear_layer = Linear(in_ft=4, out_ft=12)
    >>> x = torch.rand(10, 4)  # Batch of 10 samples with 4 features each
    >>> output = linear_layer(x)
    >>> print(output.shape)  # Output will be of shape (10, 12)

    """

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        bias: bool = True,
        weight_initializer: str = "xavier_uniform",
    ):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.init_scheme = weight_initializer

        # Parameter definitions
        self.weight = Parameter(torch.Tensor(out_ft, in_ft))  # Weight matrix
        if bias:
            self.bias = Parameter(torch.Tensor(out_ft))  # Bias vector
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain: float = 1.414):
        """Resets the parameters of the layer according to the specified initializer.

        Parameters
        ----------
        gain : float, optional
            The gain for weight initialization. Default is 1.414.

        Raises
        ------
        RuntimeError
            If the specified weight initializer is not supported.
        """
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
                f"Weight initializer '{self.init_scheme}' is not supported"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, in_ft), where N is the batch size.

        Returns
        -------
        Tensor
            Output tensor of shape (N, out_ft).
        """
        return F.linear(x, self.weight, self.bias)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}({self.in_ft}, "
            f"{self.out_ft}, bias={self.bias is not None})"
        )


class TensorLinear(nn.Module):
    r"""3D linear layer that applies the transformation: \( y = xW + b \)

    The input tensor is expected to have three dimensions, where the second
    dimension corresponds to the features, and the last dimension corresponds to
    the channel depth.

    Parameters
    ----------
    in_ft : int
        Number of input features.
    out_ft : int
        Number of output features.
    in_ch_ft : int
        Number of input channels (depth).
    out_ch_ft : int
        Number of output channels (depth).
    bias : bool, optional
        If True, the layer will learn an additive bias. Default is True.
    weight_initializer : str, optional
        The initializer for the weights. Options are: 
        'xavier_uniform', 'xavier_normal', or 'uniform'. Default is 'xavier_uniform'.

    Examples
    --------
    >>> tensor_linear_layer = TensorLinear(in_ft=4, out_ft=12, in_ch_ft=20, out_ch_ft=30)
    >>> x = torch.rand(10, 4, 20)  # Batch of 10 samples with 4 features and 20 channels each
    >>> output = tensor_linear_layer(x)
    >>> print(output.shape)  # Output will be of shape (10, 12, 30)

    """

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        in_ch_ft: int,
        out_ch_ft: int,
        bias: bool = True,
        weight_initializer: str = "xavier_uniform",
    ):
        super().__init__()
        self.in_ft = in_ft
        self.out_ft = out_ft
        self.in_ch_ft = in_ch_ft
        self.out_ch_ft = out_ch_ft
        self.init_scheme = weight_initializer

        # Parameter definitions
        self.weight = Parameter(torch.Tensor(in_ft, out_ft, out_ch_ft))  # Weight tensor
        if bias:
            self.bias = Parameter(torch.Tensor(out_ch_ft))  # Bias tensor
        else:
            self.register_parameter("bias", None)

        self.reset_parameters()

    @torch.no_grad()
    def reset_parameters(self, gain: float = 1.414):
        """Resets the parameters of the layer according to the specified initializer.

        Parameters
        ----------
        gain : float, optional
            The gain for weight initialization. Default is 1.414.

        Raises
        ------
        RuntimeError
            If the specified weight initializer is not supported.
        """
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
                f"Weight initializer '{self.init_scheme}' is not supported"
            )

    def forward(self, x: Tensor) -> Tensor:
        """Forward pass through the layer.

        Parameters
        ----------
        x : Tensor
            Input tensor of shape (N, in_ft, in_ch_ft), where N is the batch size.

        Returns
        -------
        Tensor
            Output tensor of shape (N, out_ft, out_ch_ft).
        
        Raises
        ------
        ValueError
            If the shape of the input tensor does not match the expected dimensions.
        """
        assert len(x.shape) == 3, "Input tensor must have 3 dimensions (batch, features, channels)."
        
        if x.shape[1] != self.in_ft:
            raise ValueError(
                "The feature dimension in the input tensor must "
                + f"equal to in_features in the model, got {x.shape[1]} and {self.in_ft}."
            )

        if x.shape[-1] != self.in_ch_ft:
            raise ValueError(
                "The depth channel in the input tensor must be equal to the "
                + f"number of input channels in the model, got {x.shape[-1]} and {self.in_ch_ft}."
            )

        return torch.einsum("vid,iot->vot", x, self.weight) + self.bias
