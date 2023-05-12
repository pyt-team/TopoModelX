"""Convolutional layer for message passing."""

import torch

from topomodelx.base.message_passing import _MessagePassing


class MessagePassingConv(_MessagePassing):
    """Message passing: steps 1, 2, and 3.

    Builds the message passing route given by one neighborhood matrix.
    Includes an option for a message-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    neighborhood : torch.sparse
        Neighborhood matrix.
    inter_agg_norm : bool
        Whether to normalize the aggregated message by the neighborhood size.
    update_on_message : string
        Update method to apply to message.
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        neighborhood,
        inter_agg_norm=False,
        update_on_message=None,
        initialization="xavier_uniform",
    ):
        super().__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            update_on_message=update_on_message,
            initialization=initialization,
        )
        self.neighborhood = neighborhood
        self.inter_agg_norm = inter_agg_norm
        self.update_on_message = update_on_message

    def forward(self, x):
        """Forward computation.

        Parameters
        ----------
        x: torch.tensor
            shape=[n_cells, in_channels]
            Input features on the cells.
        """
        weighted_x = torch.mm(x, self.weight)
        message = torch.mm(self.neighborhood, weighted_x)
        if self.inter_agg_norm:
            neighborhood_size = torch.sum(self.neighborhood.to_dense(), dim=1)
            message = torch.einsum("i,ij->ij", 1 / neighborhood_size, message)
        if self.update_on_message is not None:
            message = self.update(message)

        return message
