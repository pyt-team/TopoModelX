"""Convolutional layer for message passing."""

import torch
import numpy as np
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from scipy.sparse import coo_matrix
from multiprocessing import Pool

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

        # TODO: (+efficiency) We are going through the same range in each of the init
        self.weight = torch.nn.ParameterList([Parameter(torch.Tensor(self.source_in_channels, self.source_out_channels))
                                              for _ in range(self.m_hop)])
        self.att_weight = torch.nn.ParameterList([Parameter(torch.Tensor(2 * self.source_out_channels, 1))

                                                  for _ in range(self.m_hop)])
        self.negative_slope = negative_slope
        self.softmax = softmax

        for w, a in zip(self.weight, self.att_weight):
            self.reset_parameters(w, a)

    def reset_parameters(self, weight, att_weight, gain=1.414):
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
            torch.nn.init.xavier_uniform_(weight, gain=gain)
            torch.nn.init.xavier_uniform_(att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(weight, gain=gain)
            torch.nn.init.xavier_normal_(att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(self, message, A_p, a_p):
        n_messages = message.shape[0]
        A_p.coalesce()
        source_index_i, source_index_j = A_p.indices()
        s_to_s = torch.cat([message[source_index_i], message[source_index_j]], dim=1)
        e_p = torch.sparse_coo_tensor(
            indices=torch.tensor([source_index_i.tolist(), source_index_j.tolist()]),
            values=F.leaky_relu(torch.matmul(s_to_s, a_p),
                                negative_slope=self.negative_slope).squeeze(1),
            size=(n_messages, n_messages)
        )
        att_p = torch.sparse.softmax(e_p, dim=1) if self.softmax else self.sparse_row_norm(e_p)
        return att_p

    # TODO: parallelize
    # TODO: test
    def forward(self, x_source, neighborhood):
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,

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

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """

        message = [torch.mm(x_source, w) for w in self.weight]  # [m-hop, n_source_cells, d_t_out]
        result = torch.eye(x_source.shape[0]).to_sparse_coo()
        print(type(neighborhood))
        print(type(result))
        neighborhood = [result := torch.sparse.mm(neighborhood, result) for _ in range(self.m_hop)]

        # TODO: parallelize?
        # with Pool() as pool:
        # att_p = pool.map(self.attention, message)

        att = [self.attention(m_p, A_p, a_p) for m_p, A_p, a_p in zip(message, neighborhood, self.att_weight)]

        def sparse_hadamard(A_p, att_p):
            return torch.sparse_coo_tensor(
                indices=A_p.indices(),
                values=att_p.values() * A_p.values(),
                size=A_p.shape,
            )

        neighborhood = [sparse_hadamard(A_p, att_p) for A_p, att_p in zip(neighborhood, att)]

        message = [torch.mm(n_p, m_p) for n_p, m_p in zip(neighborhood, message)]

        result = torch.zeros_like(message[0])

        for m_p in message:
            result += m_p

        return self.update(result)

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


# TRASH CODE
"""
    def forward(self, x_source, neighborhood):
        Forward pass.

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
       

        message = [torch.mm(x_source, w) for w in self.weight]  # [m-hop, n_source_cells, d_t_out]

        neighborhood = neighborhood.coalesce()

        self.source_index_i = []
        self.source_index_j = []

        for p in range(self.m_hop):
            neighborhood = torch.sparse.mm(neighborhood, neighborhood)
            source_index_i, source_index_j = neighborhood._indices()
            self.source_index_i.append(source_index_i)
            self.source_index_j.append(source_index_j)
        
        att_p = [self.attention(m,) for m in message]
        
        attention = self.attention(message)

        neighborhood = [torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention[p].values() * neighborhood.values(),
            size=neighborhood.shape,
        ) for p in range(self.m_hop)]

        
        message = torch.mm(neighborhood, message)

        return self.update(message)
"""

"""
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

       return att_p"""
