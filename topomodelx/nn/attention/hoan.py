from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.util import sp_matmul, sp_softmax

"""
Two main layers are implemented here that are functionally
identitcal and represent Higher Order Attention Networks (HOANs).
The implementation is also valid for regular cell complexes/simplicial complex and hypergraphs.
These layers are implemented in the following classes
HigherOrderAttentionLayer and SparseSimplicialAttentionLayer
(1) The HigherOrderAttentionLayer class is dense implementation of a HOAN.
(2) The SparseHigherOrderAttentionLayer class is a sparse
implementation of a HOAN.
"""


class HigherOrderAttentionLayer(nn.Module):
    """
    HigherOrder Attention Layer
        Introduction:
         Matheamtically higher order attention layer has the form :
             * If A is symmetric
             phi( A . att * X * W ) where
                 A is a symmetric cochain operator A:C^i -> C^i.
                   The element of this matrix are assumed
                   to be from the set {0,1}.
                 att is an attention layer that has the
                   same shape as the operator A.
                 X is the input feature vector that belongs
                   to the cochain space C^i.
                 W is a trainable parameter matrix.
                 phi is non-linearity.
                 operation * is matrix multiplication
                 operation . is element-wise multiplication
             * If A is Asymmetric
                 phi( A . att_s2t * X_s * W_s2t )
                 phi( A.T . att_t2s * X_t * W_t2s )
             where A is a symmetric cochain operator A:C^s -> C^t
                   (Dense rep. of A is of shape
                    [num_target_cell, num_source_cell] ).
                   The element of this matrix are assumed to
                   be from the set {0,1}.
                   att_s2t is an attention matrix that has
                   the same shape as the operator A,
                   [num_target_cell, num_source_cell].
                   att_t2s is an attention matrix that has
                   the same shape as the tranpose of operator A,
                   [num_source_cell,num_target_cell].
                   X_s is the input feature vector that
                   belongs to the cochain space C^s
                   X_t is the input feature vector that
                   belongs to the cochain space C^t
                   W_s2t, and W_t2s are a trainable parameter matrix.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        dropout=0.2,
        alpha=0.1,
        concat=True,
        bias=True,
    ):
        super(HigherOrderAttentionLayer, self).__init__()

        self.source_in_features = source_in_features
        self.source_out_features = source_out_features

        self.target_in_features = target_in_features
        self.target_out_features = target_out_features

        self.alpha = alpha
        self.concat = concat
        self.dropout = dropout

        self.Ws = nn.Parameter(
            torch.empty(size=(source_in_features, target_out_features))
        )
        self.Wt = nn.Parameter(
            torch.empty(size=(target_in_features, source_out_features))
        )

        self.a1 = nn.Parameter(
            torch.empty(size=(source_out_features + target_out_features, 1))
        )

        # TODO: attension on higher order may occur
        # between simplices/cells of different dimensions.
        # Hence, maybe attension s->t should be trained seperatly from t->s
        # self.a2 = nn.Parameter(torch.empty(size=(source_out_features
        #                                        +target_out_features, 1)))

        self.leakyrelu = nn.LeakyReLU(self.alpha)

        if bias:
            self.bias_s = nn.Parameter(torch.FloatTensor(target_out_features))
            self.bias_t = nn.Parameter(torch.FloatTensor(source_out_features))
        else:
            self.bias_s, self.bias_t = None, None
        self.reset_parameters()

    def reset_parameters(self, gain=1.414):  # TODO: support + methods.

        nn.init.xavier_uniform_(self.Ws.data, gain=gain)
        nn.init.xavier_uniform_(self.Wt.data, gain=gain)

        nn.init.xavier_uniform_(self.a1.data, gain=gain)
        # nn.init.xavier_uniform_(self.a2.data, gain=gain)

        if self.bias_s is not None:
            self.bias_s.data.fill_(0)
            self.bias_t.data.fill_(0)

    def forward(self, hs, ht, A_opt):
        """
        Args:
            hs. A torch tensor of size
                [num_source_cell, num_source_in_features]
            ht. A torch tensor of size
                [num_target_cell, num_target_in_features]
            A_opt. A torch tensor of size
                [num_target_cell, num_source_cell].
                Typically this can be a boundary matrix
                when the operator is asymmetrical,
                the adjacency, laplacian matrix when
                the operator is symmetric.
            -A_opt is asymmetric:
                the matrix A_opt is not symmetric,
                one should think about this operator
                as a function that moves features
                hs (source) to feature ht ( target )
            -A_opt is symmetric:
                when the matrix A_opt is symmtric,
                one should think about the this operator
                as a function that updates the signal
                or features on eeah source cell.
                Here source and target cells are the same.
                In other words, the source and the target are
                idenitical. This is similar to the graph attension
                case except here that we can have attension
                between edges, faces and higher dimensional cells/simplices.,
        Return :
            When the operator A_opt is Asymmetric
                h_prime_st. A torch tensor of size
                    [num_source_cell, num_target_out_features]
                h_prime_ts. A torch tensor of size
                    [num_target_cell, num_source_out_features]
            When the operator A_opt is symmetric
                h_prime_st. A torch tensor of size
                    [num_source_cell, num_target_out_features],
                    here num num_source_cell=num_target_cell
                h_prime_ts. None
        """
        if len(A_opt.shape) != 2:
            raise ValueError("the input matrix operator A_opt must be a 2d tensor.")

        if ht is None:
            if A_opt.shape[0] != A_opt.shape[1]:
                raise ValueError(
                    "The operator A must be symmetric when the target features are None ."
                )
            if self.target_out_features != self.source_out_features:
                raise ValueError(
                    "the target out features dimension and the source out feature dimensions must be the same when the operator A is symmetric"
                )
        else:
            if A_opt.shape[0] == A_opt.shape[1]:
                raise ValueError(
                    "The input operator is symmetric and the target"
                    + "feature is not None."
                    + "The target features must",
                    "be None when the operator A is symmetric."
                    + "Set the target feature vector, second input,"
                    + "to None and repeat the computation.",
                )

        if len(hs.shape) == 3 and hs.shape[0] != 1:
            raise Exception(" batch multiplication is not supported.")

        if len(hs.shape) == 3 and hs.shape[0] == 1:  # fix shape if needed
            warn(
                "The first input tensor has shape "
                "[1,num_in_simplices,num_features],"
                "the tensor is being reshaped to "
                "[num_in_simplices,num_features].",
                stacklevel=2,
            )

            hs = hs.squeeze(0)

        if ht is not None:
            if len(ht.shape) == 3 and ht.shape[0] != 1:
                raise Exception(" batch multiplication is not supported.")
            if len(ht.shape) == 3 and ht.shape[0] == 1:
                warn(
                    "The first tensor has shape"
                    "[1,num_in_simplices,num_features],"
                    " the tensor is being reshaped to "
                    "[num_in_simplices,num_features].",
                    stacklevel=2,
                )

                ht = ht.squeeze(0)

        if hs.shape[0] != A_opt.shape[1]:
            raise ValueError(
                " num_source_cell in the second argument tensor hs  must match A_opt.shape[1]."
            )

        if ht is not None:
            if ht.shape[0] != A_opt.shape[0]:
                # TODO write better error
                raise ValueError(
                    "num_source_cell the first argument "
                    + "tensor ht must match A_opt.shape[0]  "
                )

        hs = F.dropout(hs, self.dropout, self.training)

        Whs = hs.matmul(self.Ws)
        if ht is not None:

            ht = F.dropout(ht, self.dropout, self.training)  # dropout
            Wht = ht.matmul(self.Wt)
        else:
            Wht = None

        e_st, e_ts = self._prepare_higher_order_attentional_input(Whs, Wht)

        zero_vec1 = -9e15 * torch.ones_like(e_st)

        attention_st = torch.where(A_opt > 0, e_st, zero_vec1)

        attention_st = F.softmax(attention_st, dim=1)
        attention_st = F.dropout(attention_st, self.dropout, training=self.training)

        h_prime_st = torch.matmul(attention_st, Whs)
        if self.bias_s is not None:
            h_prime_st = h_prime_st + self.bias_s
        if ht is not None:
            zero_vec2 = -9e15 * torch.ones_like(e_ts)
            attention_ts = torch.where(A_opt.T > 0, e_ts, zero_vec2)
            attention_ts = F.softmax(attention_ts, dim=1)
            attention_ts = F.dropout(attention_ts, self.dropout, training=self.training)
            h_prime_ts = torch.matmul(attention_ts, Wht)

            if self.bias_t is not None:
                h_prime_ts = h_prime_ts + self.bias_t

            if self.concat:
                return F.elu(h_prime_st), F.elu(h_prime_ts)
            else:
                return h_prime_st, h_prime_ts

        else:
            if self.concat:
                return F.elu(h_prime_st), None
            else:
                return h_prime_st, None

    def _prepare_higher_order_attentional_input(self, Whs, Wht):
        """
        Purpose:
            compute attensions simplex/cell s <-> simplex/cell t

        Rational:
            -Assuming asymmetric cochain operator:
                When considering higher order attention
                between cells of different dimensions s and t,
                one needs signals on both the source and the
                target cells/simplicies.
                The attension matrix, between a cell with s and another
                cell t is determined by a coeff e[s,t] which represents
                the attension that is given to the cell t by the cell s.
                To consider the attenion e[t,s] one needs to look at
                the tranpose relationship between the cells
                the sources and the target
            -Assuming symmetric cochain operator:
                This case is not different from the graph case,
                the attension matrix here is symmetric and its
                size is identical the size of the input
                cochain linear operator.

        Args:
            Whs : torch tensor of shape
                [ num_in_cell , num_source_out_features ]
            Wht : torch tensor of shape
                [ num_out_cell , num_target_out_features ]

        Return:
            e_st : torch tensor of shape
                [ num_target_cell , num_source_cell ].
                Represents the attension between s -> t.
            e_ts : torch tensor of shape
                [ num_source_cell , num_target_cell ].
                Represents the attension between t -> s.
        """

        if Wht is not None:
            Whs1 = torch.matmul(Whs, self.a1[: self.target_out_features, :])

            Wht1 = torch.matmul(Wht, self.a1[self.target_out_features :, :])

            e_st = Whs1.T + Wht1

            e_ts = Whs1 + Wht1.T

            return self.leakyrelu(e_st), self.leakyrelu(e_ts)

        else:
            Whs1 = torch.matmul(Whs, self.a1[: self.source_out_features, :])

            Whs2 = torch.matmul(Whs, self.a1[self.source_out_features :, :])
            return self.leakyrelu(Whs1.T + Whs2), None


class SparseHigherOrderAttentionLayer(HigherOrderAttentionLayer):
    """
    Introduction:
         Matheamtically higher order attention layer has the form :
             * If A is symmetric
             phi( A . att * X * W ) where
                 A: is a symmetric cochain operator A:C^i -> C^i.
                     The element of this matrix are assumed to be
                     from the set {0,1}.
                 att: is an attention layer that has the same
                   shape as the operator A
                 X: is the input feature vector that
                 belongs to the cochain space C^i
                 W: is a trainable parameter matrix.
             * If A is Asymmetric then
                 phi( A . att_s2t * X_s * W_s2t )
                 phi( A.T . att_t2s * X_t * W_t2s )
                 where
                 A: is a symmetric cochain operator A:C^s -> C^t
                    (Dense rep. of A is of shape
                    [num_target_cell, num_source_cell] ).
                    The element of this matrix are assumed
                    to be from the set {0,1}.
                 att_s2t: is an attention matrix that has the same shape as
                   the operator A, [num_target_cell, num_source_cell].
                 att_t2s: is an attention matrix that has the same shape
                   as the tranpose of operator A,
                   [num_source_cell,num_target_cell].
                 X_s: is the input feature vector that
                     belongs to the cochain space C^s
                 X_t: is the input feature vector that
                     belongs to the cochain space C^t
                 W_s2t, and W_t2s are a trainable parameter matrix.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        dropout=0.1,
        alpha=0.1,
        concat=True,
        bias=True,
    ):
        super(SparseHigherOrderAttentionLayer, self).__init__(
            source_in_features,
            target_in_features,
            source_out_features,
            target_out_features,
            dropout,
            alpha,
            concat,
            bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()

    def forward(self, hs, ht, operator_list, operator_symmetry=False):
        """
        Args:
             -hs. A torch tensor of size
                 [num_source_cell, num_source_in_features]
             -ht. A torch tensor of size
                 [num_target_cell, num_target_in_features]
             -operator_list. A torch tensor of size
                 represents the cochain map C^s -> C^t.
                 Operator_list expect a sparse rep of this
                  operator and in this case we use the sparse rep to be
                  a torch tensor of the shape [2, K] where K is the
                  number of non-zero elements dense matrix corresoponds
                  to the sparse array.
                  First row in this matrix is the rows_indices
                  of the sparse rep and the second row is
                  the column_indices.
                  In other words, if there is a message between
                  cell i and a cell j then there the sparse representation
                  that stores the indices i and j.
                  For exampl the dense operator A_opt= tensor([[1., 1., 0.],
                                                              [0., 1., 1.]])
                  can be written as
                  operator_list= tensor([[0, 0, 1, 1],
                                        [0, 1, 1, 2]])
                  which stores the positions of the nonzero elements in
                  the (dense representation ) of the operator_list
                  Typically operator_list can be a boundary matrix
                  when the operator is asymmetrical,
                  the adjacency, laplacian matrix when
                  the operator is symmetric.
             -operator_symmetry. bool, indicating
                  if the is symmetric or not.
        Return:
             When the operator A_opt is Asymmetric
                 (operator_symmetry is False):
                 -h_prime_st. A torch tensor of size
                     [num_source_cell, num_target_out_features]
                 -h_prime_ts. A torch tensor of size
                     [num_target_cell, num_source_out_features]
             When the operator A_opt is symmetric (operator_symmetry is True):
                 -h_prime_st. A torch tensor of size
                     [num_source_cell, num_target_out_features],
                     here num num_source_cell=num_target_cell
                 -h_prime_ts. None

        """

        if ht is None:
            if self.target_out_features != self.source_out_features:
                raise ValueError(
                    "The target out features dimension and the source "
                    + "out feature dimensions must be the same when the "
                    + "operator A is symmetric."
                )
            if operator_symmetry is not True:
                raise ValueError(
                    "operator_symmetry must be True when the "
                    "second input argument tensor is None."
                )

        if operator_symmetry:
            if ht is not None:
                raise ValueError(
                    "When operator_symmetry is true, the "
                    "second input argument tensor must be None."
                )
            num_cells = hs.shape[0]
        else:
            num_cells = ht.shape[0]

        target, source = operator_list
        hs = F.dropout(hs, self.dropout, self.training)
        Whs = torch.matmul(hs, self.Ws)

        if ht is not None:

            ht = F.dropout(ht, self.dropout, self.training)
            Wht = torch.matmul(ht, self.Wt)
            a_input_1 = torch.cat([Whs[source], Wht[target]], dim=1)

        else:

            Wht = None
            a_input_1 = torch.cat([Whs[source], Whs[target]], dim=1)

        e_st = F.leaky_relu(torch.matmul(a_input_1, self.a1), negative_slope=self.alpha)
        attention_st = sp_softmax(
            operator_list, torch.FloatTensor(e_st), e_st.size(0), dim=1
        )
        attention_st = F.dropout(attention_st, self.dropout, training=self.training)

        h_prime_st = sp_matmul(
            operator_list,
            attention_st,
            Whs,
            (num_cells, self.target_out_features),
            dim=1,
        )

        if self.bias_s is not None:
            h_prime_st = h_prime_st + self.bias_s

        if operator_symmetry is False:

            a_input_2 = torch.cat([Wht[target], Whs[source]], dim=1)
            a_rev = torch.cat(
                [
                    self.a1[self.source_out_features :, :],
                    self.a1[: self.source_out_features, :],
                ],
                dim=0,
            )
            e_ts = torch.matmul(a_input_2, a_rev)
            attention_ts = sp_softmax(
                operator_list, torch.FloatTensor(e_ts), e_ts.size(0), dim=0
            )
            attention_ts = F.dropout(attention_ts, self.dropout, training=self.training)
            h_prime_ts = sp_matmul(
                operator_list,
                attention_ts,
                Wht,
                (hs.shape[0], self.source_out_features),
                dim=0,
            )
            if self.bias_t is not None:
                h_prime_ts = h_prime_ts + self.bias_t

        if operator_symmetry:
            if self.concat:
                return F.elu(h_prime_st), None
            else:
                return h_prime_st, None
        else:
            if self.concat:
                return F.elu(h_prime_st), F.elu(h_prime_ts)
            else:
                return h_prime_st, h_prime_ts
