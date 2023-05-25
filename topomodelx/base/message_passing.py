"""Message passing module."""

import torch

from topomodelx.utils.scatter import scatter


class MessagePassing(torch.nn.Module):
    """MessagePassing.

    This class abstractly defines the mechanisms of message passing.

    Notes
    -----
    This class is not meant to be instantiated directly.
    Instead, it is meant to be inherited by other classes that will
    effectively define the message passing mechanism.

    For example, this class does not have trainable weights.
    The classes that inherit from it will define these weights.

    Parameters
    ----------
    aggr_func : string
        Aggregation function to use.
    att : bool
        Whether to use attention.
    initialization : string
        Initialization method for the weights of the layer.
    """

    def __init__(
        self,
        aggr_func="sum",
        att=False,
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by children classes of
        MessagePassing that will define their weights.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)
            if self.att:
                torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight, gain=gain)
            if self.att:
                torch.nn.init.xavier_normal_(self.att_weight, gain=gain)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.initialization}' is not supported"
            )

    def attention_between_cells_of_different_ranks(self, x_source, x_target):
        """Compute attention weights for messages between cells of different ranks.

        Notes
        -----
        The attention weights are given in the order of the (source, target) pairs
        that correspond to non-zero coefficients in the neighborhood matrix.

        For this reason, the attention weights are computed only after
        self.target_index_i and self.source_index_j.

        This mechanism works for neighborhood that between cells of same ranks.
        In that case, we note that the neighborhood matrix is square.

        Parameters
        ----------
        x_source : torch.tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.tensor, shape=[n_target_cells, in_channels]
            Input features on target cells.
            Assumes that all target cells have the same rank s.

        Returns
        -------
        _ : torch.tensor, shape = [n_messages, 1]
            Attention weights.
        """
        x_per_source_target_pair = torch.cat(
            [x_source[self.source_index_j], x_target[self.target_index_i]], dim=1
        )
        return torch.nn.functional.elu(
            torch.matmul(x_per_source_target_pair, self.att_weight)
        )

    def attention_between_cells_of_same_rank(self, x):
        """Compute attention weights for messages between cells of same rank.

        This provides a default attention method to the layer.

        Alternatively, users can choose to inherit from this class and overwrite
        this method to provide their own attention mechanism.

        Notes
        -----
        The attention weights are given in the order of the (source, target) pairs
        that correspond to non-zero coefficients in the neighborhood matrix.

        For this reason, the attention weights are computed only after
        self.target_index_i and self.source_index_j.

        This mechanism works for neighborhood that between cells of same ranks.
        In that case, we note that the neighborhood matrix is square.

        Parameters
        ----------
        x : torch.tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.tensor
            shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_per_source_target_pair = torch.cat(
            [x[self.source_index_j], x[self.target_index_i]], dim=1
        )
        return torch.nn.functional.elu(
            torch.matmul(x_per_source_target_pair, self.att_weight)
        )

    def attention(self, x):
        """Compute attention weights for messages between cells of same rank.

        This provides a default attention method to the layer.

        Alternatively, users can choose to inherit from this class and overwrite
        this method to provide their own attention mechanism.

        For example, they can choose to use the method
        attention_between_cells_of_different_ranks instead.

        Parameters
        ----------
        x : torch.tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.tensor, shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """
        return self.attention_between_cells_of_same_rank(x)

    def propagate(self, x, neighborhood):
        """Propagate messages from source cells to target cells.

        This only propagates the values in x using the neighborhood matrix.

        There is no weight in this function.

        Parameters
        ----------
        x : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        assert isinstance(x, torch.Tensor)
        assert isinstance(neighborhood, torch.Tensor)
        assert neighborhood.ndim == 2

        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()

        if self.att:
            if neighborhood.shape[0] != neighborhood.shape[1]:
                raise RuntimeError(
                    "Use attention mechanism between cells of different ranks."
                )
            attention_values = self.attention(x)

        x = self.message(x)
        x = self.sparsify_message(x)
        neighborhood_values = neighborhood.values()
        if self.att:
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x = neighborhood_values.view(-1, 1) * x
        x = self.aggregate(x)
        return x

    def sparsify_message(self, x):
        """Construct message from source cells, indexed by j.

        This extracts the features in x that are involved in messages,
        i.e. that correspond to non-zero entries of the neighborhood matrix.

        Parameters
        ----------
        x : Tensor, shape=[..., n_cells, channels]
            Features (potentially weighted or transformed) on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : Tensor, shape=[..., n_messages, channels]
            Values of x that are only at indexes j, i.e. that
            are on the source cells that are effectively involved in messages.
            We can have n_messages > n_cells, if the same cells are involved in
            several messages.
        """
        source_index_j = self.source_index_j
        return x.index_select(-2, source_index_j)

    def message(self, x):
        """Construct message from source cells with weights.

        Parameters
        ----------
        x : Tensor, shape=[..., n_source_cells, channels]
            Features (potentially weighted or transformed) on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : Tensor, shape=[..., n_source_cells, channels]
            Weighted features on source cells of rank r.
        """
        pass

    def get_x_i(self, x):
        """Get value in tensor x at index self.target_index_i.

        Note that index i is a tuple of indices, that
        represent the indices of the target cells.

        Parameters
        ----------
        x : Tensor, shape=[..., n_source_cells, channels]
            Features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : Tensor, shape=[..., n_messages, channels]
            Values of x that are only at indexes i, which correspond
            to values on the target cells that receive messages.
        """
        return x.index_select(-2, self.target_index_i)

    def aggregate(self, x):
        """Aggregate values in input tensor.

        A given target cell can receive several messages from several source cells.
        This function aggregates these messages into a single feature per target cell.

        Parameters
        ----------
        x : Tensor, shape=[..., n_messages, out_channels]
            Features associated with each message, i.e. each pair of source-target
            cells.

        Returns
        -------
        _ : Tensor, shape=[...,  n_target_cells, out_channels]
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        out = aggr(x, self.target_index_i, 0)
        return out

    def forward(self, x, neighborhood):
        r"""Run the forward pass of the module.

        Parameters
        ----------
        x : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all source cells have the same rank s.
        """
        return self.propagate(x, neighborhood)
