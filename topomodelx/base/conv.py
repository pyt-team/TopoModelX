"""Convolutional layer for x passing."""

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


class Conv(MessagePassing):
    """Message passing: steps 1, 2, and 3.

    Builds the x passing route given by one neighborhood matrix.
    Includes an option for a x-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool
        Whether to normalize the aggregated x by the neighborhood size.
    update_func : string
        Update method to apply to x.
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm=False,
        update_func=None,
        initialization="xavier_uniform",
    ):
        super().__init__(
            update_func=update_func,
            initialization=initialization,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        self.reset_parameters()

    def forward(self, x, neighborhood):
        """Forward computation.

        Parameters
        ----------
        x: torch.tensor
            shape=[n_cells, in_channels]
            Input features on the cells.
        neighborhood : torch.sparse
            Neighborhood matrix.

        Returns
        -------
        _ : torch.tensor
            shape=[n_cells, out_channels]
            Output features on the cells.
        """
        x = torch.mm(x, self.weight)
        x = torch.mm(neighborhood, x)
        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x = torch.einsum("i,ij->ij", 1 / neighborhood_size, x)
        if self.update_func is not None:
            x = self.update(x)
        return x
