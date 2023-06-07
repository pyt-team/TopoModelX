"""Convolutional layer for message passing."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix

from topomodelx.base.message_passing import MessagePassing


class CCAT(MessagePassing):
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
            d_s_in,
            d_s_out,
            d_t_in,
            d_t_out,
            negative_slope,
            aggr_norm=False,
            update_func=None,
            initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.d_s_in, self.d_s_out = d_s_in, d_s_out
        self.d_t_in, self.d_t_out = d_t_in, d_t_out

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.w_s = Parameter(torch.Tensor(self.d_s_in, self.d_t_out))
        self.w_t = Parameter(torch.Tensor(self.d_t_in, self.d_s_out))

        self.att_weight = Parameter(torch.Tensor(self.d_t_out + self.d_s_out, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

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

            if self.att:
                torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.w_s, gain=gain)
            torch.nn.init.xavier_normal_(self.w_t, gain=gain)
            if self.att:
                torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(self, x_source, x_target=None):  # TODO: Arreglar declaración de parámetros
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

        s_message = x_source
        t_message = x_target

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
                torch.matmul(t_to_s, torch.cat([self.att_weight[self.d_t_out:], self.att_weight[:self.d_t_out]])),
                negative_slope=self.negative_slope).squeeze(1),
            size=(t_message.shape[0], s_message.shape[0])
        )

        # TODO: Preguntar si en la atención deberíamos usar softmax
        # Compute the sum along dimension 1 and reshape the result

        e_sum = torch.sparse.sum(e,dim=1)
        f_sum = torch.sparse.sum(f, dim=1)

        e_values = e._values() / e_sum.to_dense()[e._indices()[0]]
        f_values = f._values() / f_sum.to_dense()[f._indices()[0]]

        e = torch.sparse_coo_tensor(e._indices(), e_values, e.shape)
        f = torch.sparse_coo_tensor(f._indices(), f_values, f.shape)

        """e_sum = torch.sum(e.to_dense(), dim=1)
        e_inv = torch.diag(torch.reciprocal(e_sum))
        e = torch.matmul(e_inv, e.to_dense())
        e[torch.isnan(e)] = 0
        e = e.to_sparse_coo()

        f_sum = torch.sum(f.to_dense(), dim=1)
        f_inv = torch.diag(torch.reciprocal(f_sum))
        f = torch.matmul(f_inv, f.to_dense())
        f[torch.isnan(f)] = 0
        f = f.to_sparse_coo()"""

        #e = torch.sparse.softmax(e, dim=1)
        #f = torch.sparse.softmax(f, dim=1)"""

        return e.coalesce(),f.coalesce()

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

        s_message = torch.mm(x_source, self.w_s)  # [n_source_cells, d_t_out]
        t_message = torch.mm(x_target, self.w_t)  # [n_target_cells, d_s_out]

        neighborhood_s = neighborhood.coalesce()  # TODO: Qué hace coalesce
        neighborhood_t = neighborhood.t().coalesce()

        self.source_index_i, self.target_index_j = neighborhood_s.indices()
        self.target_index_i, self.source_index_j = neighborhood_t.indices()

        s_t_attention, t_s_attention = self.attention(s_message, t_message)

        neighborhood_s_t = torch.sparse_coo_tensor(
            indices=neighborhood_s.indices(),
            values=s_t_attention.values() * neighborhood_s.values(),
            size=neighborhood.shape,
        )

        print(neighborhood_s_t.to_dense())

        neighborhood_t_s = torch.sparse_coo_tensor(
            indices=neighborhood_t.indices(),
            values=t_s_attention.values() * neighborhood_t.values(),
            size=neighborhood_t.shape,
        )

        message_on_source = torch.mm(neighborhood_s_t, t_message)
        message_on_target = torch.mm(neighborhood_t_s, s_message)

        if self.update_func == "sigmoid":
            message_on_target = torch.sigmoid(message_on_target)
            message_on_source = torch.sigmoid(message_on_source)
        elif self.update_func == "relu":
            message_on_target = torch.nn.functional.relu(message_on_target)
            message_on_source = torch.nn.functional.relu(message_on_source)

        return message_on_source, message_on_target

    def divide_by_row_sum(self, sparse_coo):
        # Converting COO to CSR format to perform row-wise operations
        sparse_csr = sparse_coo.to_sparse_csr()

        # Sum across each row and reshape to a column vector
        row_sums = sparse_csr.sum(axis=1)

        # Take inverse of row sums to avoid division (division is more costly)
        # A small constant is added to avoid division by zero.
        row_sums_inv = 1.0 / (row_sums + 1e-10)

        # Diagonal matrix of inverses
        row_sums_inv_diag = coo_matrix((row_sums_inv.flatten(), (range(len(row_sums)), range(len(row_sums)))))

        # Multiply original sparse matrix with the diagonal one (equivalent to dividing each element by the row sum)
        result = row_sums_inv_diag.dot(sparse_csr)

        return result