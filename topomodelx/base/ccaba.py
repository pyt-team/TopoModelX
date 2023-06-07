"""Convolutional layer for message passing."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix

from topomodelx.base.message_passing import MessagePassing


class CCABA(MessagePassing):

    def __init__(
            self,
            d_s_in,
            d_s_out,
            negative_slope,
            aggr_norm=False,
            update_func=None,
            initialization="xavier_uniform",
    ):

        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.d_s_in = d_s_in
        self.d_s_out = d_s_out

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = Parameter(torch.Tensor(self.d_s_in, self.d_s_out))

        self.att_weight = Parameter(torch.Tensor(2*self.d_s_out, 1))
        self.negative_slope = negative_slope

        self.reset_parameters()

    def attention(self, x_source, x_target=None):  # TODO: Arreglar declaración de parámetros

        message = x_source
        n_messages = message.shape[0]

        s_to_s = torch.cat(
            [message[self.source_index_i], message[self.source_index_j]], dim=1
        )

        e = torch.sparse_coo_tensor(
            indices=torch.tensor([self.source_index_i.tolist(), self.source_index_j.tolist()]),
            values=F.leaky_relu(torch.matmul(s_to_s, self.att_weight), negative_slope=self.negative_slope).squeeze(1),
            size=(n_messages, n_messages)
        )

        return self.sparse_row_norm(e)

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

        message = torch.mm(x_source, self.weight)  # [n_source_cells, d_t_out]

        neighborhood = neighborhood.coalesce()

        self.source_index_i, self.source_index_j = neighborhood.indices()

        attention = self.attention(message)

        neighborhood = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values= attention.values() * neighborhood.values(),
            size=neighborhood.shape,
        )

        message = torch.mm(neighborhood, message)

        if self.update_func == "sigmoid":
            message = torch.sigmoid(message)
        elif self.update_func == "relu":
            message = F.relu(message)

        return message

    def sparse_row_norm(self, sparse_tensor):
        row_sum = torch.sparse.sum(sparse_tensor, dim=1)
        values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
        sparse_tensor = torch.sparse_coo_tensor(sparse_tensor._indices(), values, sparse_tensor.shape)
        return sparse_tensor.coalesce()