"""Convolutional layer for message passing."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix

from topomodelx.base.message_passing import MessagePassing


class HBNS(MessagePassing):
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
            source_in_channels,
            source_out_channels,
            target_in_channels,
            target_out_channels,
            negative_slope,
            softmax=False,  # TODO implementar
            aggr_norm=False, # Todo quizá quitar
            update_func=None,
            initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.source_in_channels, self.source_out_channels = source_in_channels, source_out_channels
        self.target_in_channels, self.target_out_channels = target_in_channels, target_out_channels

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.w_s = Parameter(torch.Tensor(self.source_in_channels, self.target_out_channels))
        self.w_t = Parameter(torch.Tensor(self.target_in_channels, self.source_out_channels))

        self.att_weight = Parameter(torch.Tensor(self.target_out_channels + self.source_out_channels, 1))
        self.negative_slope = negative_slope

        self.softmax = softmax

        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        """Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.w_s, gain=gain)
            torch.nn.init.xavier_uniform_(self.w_t, gain=gain)
            torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.w_s, gain=gain)
            torch.nn.init.xavier_normal_(self.w_t, gain=gain)
            torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(self, message_on_source, message_on_target):
        if self.update_func == "sigmoid":
            message_on_source = torch.sigmoid(message_on_source)
            message_on_target = torch.sigmoid(message_on_target)
        elif self.update_func == "relu":
            message_on_source = torch.nn.functional.relu(message_on_source)
            message_on_target = torch.nn.functional.relu(message_on_target)

        return message_on_source, message_on_target

    def attention(self, s_message, t_message):
        """Compute attention weights for messages.

        This provides a default attention function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        Details in [H23]_, Definition of "Attention Higher-Order Message Passing".

        Parameters
        ----------
        x_source : torch.Tensor, shatpe=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape=[n_target_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.Tensor, shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """

        s_to_t = torch.cat(
            [s_message[self.source_index_i], t_message[self.target_index_j]], dim=1
        )

        t_to_s = torch.cat(
            [t_message[self.target_index_i], s_message[self.source_index_j]], dim=1
        )

        e = torch.sparse_coo_tensor(
            indices=torch.tensor([self.source_index_i.tolist(), self.target_index_j.tolist()]),
            values=F.leaky_relu(torch.matmul(s_to_t, self.att_weight), negative_slope=self.negative_slope).squeeze(1),
            size=(s_message.shape[0], t_message.shape[0])
        )

        f = torch.sparse_coo_tensor(
            indices=torch.tensor([self.target_index_i.tolist(), self.source_index_j.tolist()]),
            values=F.leaky_relu(
                torch.matmul(t_to_s, torch.cat(
                    [self.att_weight[self.target_out_channels:], self.att_weight[:self.target_out_channels]])),
                negative_slope=self.negative_slope).squeeze(1),
            size=(t_message.shape[0], s_message.shape[0])
        )
        if self.softmax:
            return torch.sparse.softmax(e, dim=1), torch.sparse.softmax(f, dim=1)
        return self.sparse_row_norm(e), self.sparse_row_norm(f)


    def forward(self, x_source, x_target, neighborhood):
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

        s_message = torch.mm(x_source, self.w_s)  # [n_source_cells, d_t_out]
        t_message = torch.mm(x_target, self.w_t)  # [n_target_cells, d_s_out]

        neighborhood_s = neighborhood.coalesce()
        neighborhood_t = neighborhood.t().coalesce()

        self.source_index_i, self.target_index_j = neighborhood_s.indices()
        self.target_index_i, self.source_index_j = neighborhood_t.indices()

        s_t_attention, t_s_attention = self.attention(s_message, t_message)

        neighborhood_s_t = torch.sparse_coo_tensor(
            indices=neighborhood_s.indices(),
            values=s_t_attention.values() * neighborhood_s.values(),
            size=neighborhood.shape,
        )

        neighborhood_t_s = torch.sparse_coo_tensor(
            indices=neighborhood_t.indices(),
            values=t_s_attention.values() * neighborhood_t.values(),
            size=neighborhood_t.shape,
        )

        message_on_source = torch.mm(neighborhood_s_t, t_message)
        message_on_target = torch.mm(neighborhood_t_s, s_message)

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)

    def sparse_row_norm(self, sparse_tensor):
        row_sum = torch.sparse.sum(sparse_tensor, dim=1)
        values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
        sparse_tensor = torch.sparse_coo_tensor(sparse_tensor._indices(), values, sparse_tensor.shape)
        return sparse_tensor.coalesce()
