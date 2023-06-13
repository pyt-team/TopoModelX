"""Convolutional layer for message passing."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix

from topomodelx.base.message_passing import MessagePassing


class CCABA(MessagePassing):

    def __init__(
            self,
            source_in_channels,
            source_out_channels,
            negative_slope,
            softmax=False,
            m_hop=1,
            aggr_norm=False,
            update_func=None,
            initialization="xavier_uniform",
    ):

        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels

        self.m_hop = m_hop

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.weight = torch.nn.ParameterList([Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))
                                              for _ in range(self.m_hop)])

        # Add a list of parameters
        self.att_weight = torch.nn.ParameterList([Parameter(torch.Tensor(2 * self.source_out_channels, 1))
                                                  for _ in range(self.m_hop)])
        self.negative_slope = negative_slope
        self.softmax = softmax

        self.reset_parameters()

    def attention(self, message):

        n_messages = message[0].shape[0]

        s_to_s = [torch.cat([message[p][self.source_index_i[p]], message[p][self.source_index_j[p]]], dim=1) for p in range(1, self.m_hop + 1)]

        e_p = [
            torch.sparse_coo_tensor(
                indices=torch.tensor([self.source_index_i[p].tolist(), self.source_index_j[p].tolist()]),
                values=F.leaky_relu(torch.matmul(s_to_s[p], self.att_weight[p]),
                                    negative_slope=self.negative_slope).squeeze(1),
                size=(n_messages, n_messages)
            ) for p in range(1, self.m_hop + 1)
        ]

        att_p = [torch.sparse.softmax(e, dim=1) if self.softmax else self.sparse_row_norm(e) for e in e_p]

        return att_p

    def forward(self, x_source, neighborhood):
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

        message = [torch.mm(x_source, w) for w in self.weight]  # [m-hop, n_source_cells, d_t_out]

        neighborhood = neighborhood.coalesce()

        self.source_index_i = []
        self.source_index_j = []

        for p in range(self.m_hop):
            neighborhood = torch.sparse.mm(neighborhood, neighborhood)
            source_index_i, source_index_j = neighborhood.indices()
            self.source_index_i.append(source_index_i)
            self.source_index_j.append(source_index_j)

        attention = self.attention(message)

        neighborhood = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention.values() * neighborhood.values(),
            size=neighborhood.shape,
        )

        message = torch.mm(neighborhood, message)

        return self.update(message)

    def update(self, message):
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
            return torch.sigmoid(message)
        if self.update_func == "relu":
            return torch.nn.functional.relu(message)

    # TODO This code fragment is repeated in the classes CCABI and CCABA and should be placed in a utils
    # class. However, as we do not want to change the internal implementation of the TopoModelX library for the
    # challenge, we leave the code fragment duplicated.
    def sparse_row_norm(self, sparse_tensor):
        row_sum = torch.sparse.sum(sparse_tensor, dim=1)
        values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
        sparse_tensor = torch.sparse_coo_tensor(sparse_tensor._indices(), values, sparse_tensor.shape)
        return sparse_tensor.coalesce()
