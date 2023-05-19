"""Message passing module."""
import torch
import torch.nn as nn

from topomodelx.utils.scatter import scatter


class MessagePassing(torch.nn.Module):
    """MessagePassing.

    This corresponds to Steps 1 & 2 of the 4-step scheme.
    """

    def __init__(
        self,
        aggr_func="sum",
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.aggr_func = aggr_func
        self.initialization = initialization

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        This function will be called by children classes of
        MessagePassing that will define their weights.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.

        Returns
        -------
        _ : Tensor
            Weight tensor to be initialized.
        """
        if self.initialization == "xavier_uniform":
            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.initialization == "xavier_normal":
            nn.init.xavier_normal_(self.weight, gain=gain)

        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.initialization}' is not supported"
            )

        return self.weight

    def propagate(self, x, neighborhood):
        """Propagate messages from source cells to target cells.

        This only propagates the values in x using the neighborhood matrix.

        There is no weight in this function.

        Parameters
        ----------
        x : Tensor, shape=[..., n_cells, in_channels]
            Features on all cells of a given rank.
        neighborhood : torch.sparse
            Neighborhood matrix.

        Returns
        -------
        _ : Tensor, shape=[..., n_cells, out_channels]
            Features on all cells of a given rank.
        """
        assert isinstance(x, torch.Tensor)
        assert isinstance(neighborhood, torch.Tensor)
        assert neighborhood.ndim == 2

        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        x = self.message(x)
        x = self.sparsify_message(x)
        neighborhood_values = neighborhood.values()
        x = neighborhood_values.view(-1, 1) * x
        x = self.aggregate(x)
        return x

    def sparsify_message(self, x):
        """Construct message from source cells, indexed by j.

        Parameters
        ----------
        x : Tensor, shape=[..., n_cells, out_channels]
            Features (potentially transformed and weighted)
            on all cells of a given rank.

        Returns
        -------
        _ : Tensor, shape=[..., n_cells, out_channels]
            Values of x that are only at indexes j, i.e. that
            are on the source cells.
        """
        source_index_j = self.source_index_j
        return x.index_select(-2, source_index_j)

    def message(self, x):
        """Construct message from source cells with weights.

        Parameters
        ----------
        x : Tensor, shape=[..., n_cells, in_channels]
            Features (potentially transformed and weighted)
            on all cells of a given rank.

        Returns
        -------
        _ : Tensor, shape=[..., n_cells, out_channels]
            Weighted features of all cells of a given rank.
        """
        pass

    def get_x_i(self, x):
        """Get value in tensor x at index i.

        Note that index i is a tuple of indices, that
        represent the indices of the target cells.

        Parameters
        ----------
        x : Tensor, shape=[..., n_cells, out_channels]
            Input tensor.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Values of x that are only at indexes i, i.e. values
            that are on the target cells.
        """
        return x.index_select(-2, self.target_index_i)

    def aggregate(self, x):
        """Aggregate values in input tensor.

        Parameters
        ----------
        x : Tensor, shape=[..., n_target_cells, n_source_cells, out_channels]
            Input tensor. Each target cells is receiving several messages
            from several source cells.

        Returns
        -------
        _ : Tensor, shape=[...,  n_target_cells, out_channels]
            Aggregated tensor. Each target cell has aggregated the several messages
            from several source cells.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x, self.target_index_i, 0)

    def forward(self, x, neighborhood):
        r"""Run the forward pass of the module."""
        return self.propagate(x, neighborhood)
