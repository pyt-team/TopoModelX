"""Convolutional layer for message passing."""

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


class Conv(MessagePassing):
    """Message passing: steps 1, 2, and 3.

    Builds the message passing route given by one neighborhood matrix.
    Includes an option for a x-specific update function.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    aggr_norm : bool
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : string
        Update method to apply to message.
    att : bool
        Whether to use attention.
        Optional, default: False.
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        aggr_norm=False,
        update_func=None,
        att=False,
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=att,
            initialization=initialization,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    2 * self.in_channels,
                )
            )

        self.reset_parameters()

    def update(self, x_target):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_target : torch.Tensor, shape=[n_target_cells, out_channels]
            Output features on target cells.

        Returns
        -------
        _ : torch.Tensor, shape=[n_target_cells, out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_target)

    def forward(self, x_source, neighborhood, x_target=None):
        """Forward computation.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_source_cells, in_channels]
            Input features on the source cells.
        neighborhood : torch.sparse
            Neighborhood matrix.

        Returns
        -------
        _ : torch.Tensor, shape=[n_cells, out_channels]
            Output features on the cells.
        """
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source)
            attention = torch.zeros_like(neighborhood).to_dense()
            attention[self.target_index_i, self.source_index_j] = attention_values
            attention = attention.to_sparse()

        x_source = torch.mm(x_source, self.weight)
        x_source = torch.mm(neighborhood, x_source)
        if self.att:
            neighborhood = torch.mul(neighborhood, attention)
        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_source = torch.einsum("i,ij->ij", 1 / neighborhood_size, x_source)
        if self.update_func is not None:
            x_source = self.update(x_source)
        return x_source
