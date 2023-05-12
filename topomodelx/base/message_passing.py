"""Message passing module."""

import torch
import torch.nn as nn
from torch.nn.parameter import Parameter

from topomodelx.utils.scatter import scatter


class _MessagePassing(torch.nn.Module):
    """_MessagePassing.

    This corresponds to Steps 1 & 2 of the 4-step scheme.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    neighborhood : torch.sparse
        Neighborhood matrix.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        update_on_message,
        initialization,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_on_message = update_on_message
        self.initialization = initialization

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.reset_parameters()

    def message(self, x, neighborhood):
        r"""Construct message from feature x on source/sender cell.

        Note that this is different from the convention
        in pytorch-geometry which uses x as the features
        that are going to be updated, i.e. on the receiver
        cells.

        Parameters
        ----------
        x : Tensor
            Features on the source cells, that is: the cells
            sending the messages.
        """
        weighted_x = torch.mm(x, self.weight)
        message = torch.mm(neighborhood, weighted_x)
        return message

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        weight : Tensor
            Weight tensor to be initialized.
        gain : float
            Gain for the weight initialization.
        """
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

    def update(self, inputs):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        inputs : array-like, shape=[n_skleton_out, out_channels]
            Features on the skeleton out.

        Returns
        -------
        _ : array-like, shape=[n_skleton_out, out_channels]
            Updated features on the skeleton out.
        """
        if self.update_on_message == "sigmoid":
            return torch.sigmoid(inputs)
        if self.update_on_message == "relu":
            return torch.nn.functional.relu(inputs)

    def forward(self, x, neighborhood):
        r"""Run the forward pass of the module."""
        x = self.message(x, neighborhood)
        if self.update_on_message is not None:
            x = self.update(x)
        return x
