"""Message passing module."""

import torch

from topomodelx.utils.scatter import scatter


class MessagePassing(torch.nn.Module):
    """MessagePassing.

    This class abstractly defines the mechanisms of message passing.
    This class is not meant to be instantiated directly.
    Instead, it is meant to be inherited by other classes that will
    effectively define the message passing mechanism.

    Note that this class does not have trainable weights.
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
                torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.initialization}' is not supported"
            )

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages between cells of same rank.

        This provides a default attention method to the layer.

        Alternatively, users can choose to inherit from this class and overwrite
        this method to provide their own attention mechanism.

        For example, they can choose to use the method
        attention_between_cells_of_different_ranks instead.

        Parameters
        ----------
        x_source : torch.tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.tensor, shape=[n_target_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.tensor, shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def message(self, x_source, x_target=None):
        """Construct message from source cells.

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : Tensor, shape=[..., n_target_cells, in_channels]
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        _ : Tensor, shape=[..., n_source_cells, channels]
            Weighted features on source cells of rank r.
        """
        return x_source

    def aggregate(self, x_message):
        """Aggregate values in input tensor.

        A given target cell can receive several messages from several source cells.
        This function aggregates these messages into a single feature per target cell.

        Parameters
        ----------
        x_messages : Tensor, shape=[..., n_messages, out_channels]
            Features associated with each message.
            One message is sent from a source cell to a target cell.

        Returns
        -------
        _ : Tensor, shape=[...,  n_target_cells, out_channels]
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x_message, self.target_index_i, 0)

    def propagate(self, x_source, neighborhood, x_target=None):
        """Propagate messages from source cells to target cells.

        This only propagates the values in x using the neighborhood matrix.

        There is no weight in this function.

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
        assert isinstance(x_source, torch.Tensor)
        assert isinstance(neighborhood, torch.Tensor)
        assert neighborhood.ndim == 2

        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        neighborhood_values = neighborhood.values()

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x_message = neighborhood_values.view(-1, 1) * x_message
        return self.aggregate(x_message)

    def forward(self, x_source, neighborhood, x_target=None):
        r"""Run the forward pass of the module.

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
            Assumes that all source cells have the same rank s.
        """
        return self.propagate(
            x_source=x_source, neighborhood=neighborhood, x_target=x_target
        )
