"""Message passing module."""

import torch

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
    intra_aggr : string
        Aggregation method.
        (Inter-neighborhood).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_aggr="sum",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intra_aggr = intra_aggr

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        pass

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
        pass

    def aggregate(self, inputs):
        """Aggregate messages from the neighborhood.

        Intra-neighborhood aggregation.
        """
        return scatter(self.intra_aggr)(inputs)

    def update(self, inputs):
        r"""Update embeddings for each cell."""
        return inputs

    def forward(self, x, neighborhood):
        r"""Run the forward pass of the module."""
        message = self.message(x, neighborhood)
        aggregated_message = self.aggregate(message)
        output = self.update(aggregated_message)

        return output
