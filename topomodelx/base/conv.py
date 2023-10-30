"""Convolutional layer for message passing."""
from typing import Literal

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


class Conv(MessagePassing):
    """Message passing: steps 1, 2, and 3.

    Builds the message passing route given by one neighborhood matrix.
    Includes an option for an x-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : {"relu", "sigmoid"}, optional
        Update method to apply to message.
    att : bool, default=False
        Whether to use attention.
    initialization : {"xavier_uniform", "xavier_normal"}, default="xavier_uniform"
        Initialization method.
    initialization_gain : float, default=1.414
        Initialization gain.
    with_linear_transform : bool, default=True
        Whether to apply a learnable linear transform.
        NB: if `False` in_channels has to be equal to out_channels.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm: bool = False,
        update_func: Literal["relu", "sigmoid", None] = None,
        att: bool = False,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
        with_linear_transform: bool = True,
    ) -> None:
        super().__init__(
            att=att,
            initialization=initialization,
            initialization_gain=initialization_gain,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = (
            Parameter(torch.Tensor(self.in_channels, self.out_channels))
            if with_linear_transform
            else None
        )

        if not with_linear_transform and in_channels != out_channels:
            raise ValueError(
                "With `linear_trainsform=False`, in_channels has to be equal to out_channels"
            )
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    2 * self.in_channels,
                )
            )

        self.reset_parameters()

    def update(self, x_message_on_target) -> torch.Tensor:
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape = (n_target_cells, out_channels)
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)
        return x_message_on_target

    def forward(self, x_source, neighborhood, x_target=None) -> torch.Tensor:
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source, x_target)
            neighborhood = torch.sparse_coo_tensor(
                indices=neighborhood.indices(),
                values=attention_values * neighborhood.values(),
                size=neighborhood.shape,
            )
        if self.weight is not None:
            x_message = torch.mm(x_source, self.weight)
        else:
            x_message = x_source
        x_message_on_target = torch.mm(neighborhood, x_message)

        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_message_on_target = torch.einsum(
                "i,ij->ij", 1 / neighborhood_size, x_message_on_target
            )

        return self.update(x_message_on_target)
