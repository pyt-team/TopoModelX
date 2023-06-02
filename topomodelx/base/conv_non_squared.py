"""Convolutional layer for message passing."""

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


class ConvNonSquared(MessagePassing):

    def __init__(
        self,
        in_channels_source,
        in_channels_target,
        out_channels_source,
        out_channels_target,
        aggr_norm=False,
        update_func=None,
        att=False,
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=att,
            initialization=initialization,
        )
        self.in_channels_source = in_channels_source
        self.in_channels_target = in_channels_target
        self.out_channels_source = out_channels_source
        self.out_channels_target = out_channels_target

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.in_channels, self.out_channels))
        if self.att:
            self.att_weight = Parameter(
                torch.Tensor(
                    self.out_channels_source + self.out_channels_target
                )
            )

        self.reset_parameters()

    def update(self, x_message_on_target, x_target=None):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape=[n_target_cells, out_channels]
            Output features on target cells.

        Returns
        -------
        _ : torch.Tensor, shape=[n_target_cells, out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)

    def forward(self, x_source, neighborhood, x_target=None):
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
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.
        x_target : Tensor, shape=[..., n_target_cells, in_channels]
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """


        '''
        if self.att:
            neighborhood = neighborhood.coalesce()
            self.target_index_i, self.source_index_j = neighborhood.indices()
            attention_values = self.attention(x_source, x_target)
            neighborhood = torch.sparse_coo_tensor(
                indices=neighborhood.indices(),
                values=attention_values * neighborhood.values(),
                size=neighborhood.shape,
            )

        x_message = torch.mm(x_source, self.weight)
        x_message_on_target = torch.mm(neighborhood, x_message)

        if self.aggr_norm:
            neighborhood_size = torch.sum(neighborhood.to_dense(), dim=1)
            x_message_on_target = torch.einsum(
                "i,ij->ij", 1 / neighborhood_size, x_message_on_target
            )

        if self.update_func is None:
            return x_message_on_target

        return self.update(x_message_on_target, x_target)
        '''
