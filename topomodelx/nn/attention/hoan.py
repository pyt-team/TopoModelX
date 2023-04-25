"""Higher Order Attention Network (HOAN) module.

Two main layers are implemented here, which are functionally
identical and represent Higher Order Attention Networks (HOANs).

These layers are implemented in the following classes:
(1) The HigherOrderAttentionLayer class:
This is a dense implementation of a HOAN.
(2) The SparseHigherOrderAttentionLayer class:
This is a sparse implementation of a HOAN.

The implementation is also valid for regular cell complexes/simplicial complex and hypergraphs.
"""

from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.util import sp_matmul, sp_softmax


class HigherOrderAttentionLayer(nn.Module):
    """Class for a Higher Order Attention Layer (HOAN).

    This is a dense implementation of a HOAN.

    Mathematically, a higher order attention layer can be defined
    in one of the following two ways, depending on whether the
    neighborhood matrix A is symmetric or not.

    * If A is symmetric:
        phi( A . att * X * W )
        where:
            - A is a symmetric cochain operator A:C^i -> C^i.
            The element of this matrix are assumed
            to be from the set {0,1}.
            - att is an attention layer that has the
            same shape as the operator A.
            - X is the input feature vector that belongs
            to the cochain space C^i.
            - W is a trainable parameter matrix.
            - phi is a non-linearity.
            - operation * is a matrix multiplication.
            - operation . is an element-wise multiplication.
    * If A is asymmetric:
            phi( A . att_s2t * X_s * W_s2t )
            phi( A.T . att_t2s * X_t * W_t2s )
        where:
            - A is a symmetric cochain operator A:C^s -> C^t
            (Dense rep. of A is of shape
            [n_target_cell, n_source_cell] ).
            The element of this matrix are assumed to
            be from the set {0,1}.
            - att_s2t is an attention matrix that has
            the same shape as the operator A,
            [n_target_cell, n_source_cell].
            - att_t2s is an attention matrix that has
            the same shape as the transpose of operator A,
            [n_source_cell,n_target_cell].
            - X_s is the input feature vector that
            belongs to the cochain space C^s.
            - X_t is the input feature vector that
            belongs to the cochain space C^t.
            - W_s2t, and W_t2s are trainable parameter matrices.

    Parameters
    ----------
    source_in_features : int
        Number of input features for source cells.
    target_in_features : int
        Number of input features for target cells.
    source_out_features : int
        Number of output features for source cells.
    target_out_features : int
        Number of output features for target cells.
    dropout : float, optional
        Dropout rate. The default is 0.2.
    alpha : float, optional
        LeakyReLU negative slope. The default is 0.1.
    concatenate : bool, optional
        Concatenate or average attention. The default is True.
    bias : bool, optional
        Use bias. The default is True.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        dropout=0.2,
        alpha=0.1,
        concatenate=True,
        bias=True,
    ):
        super(HigherOrderAttentionLayer, self).__init__()

        self.source_in_features = source_in_features
        self.source_out_features = source_out_features

        self.target_in_features = target_in_features
        self.target_out_features = target_out_features

        self.alpha = alpha
        self.concatenate = concatenate
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

        # TODO: attention on higher order may occur
        # between simplices/cells of different dimensions.
        # Hence, maybe attention s->t should be trained seperatly from t->s
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
        """Reset parameters.

        Parameters
        ----------
        gain : float, optional
            Gain. The default is 1.414.
        """
        nn.init.xavier_uniform_(self.Ws.data, gain=gain)
        nn.init.xavier_uniform_(self.Wt.data, gain=gain)

        nn.init.xavier_uniform_(self.a1.data, gain=gain)
        # nn.init.xavier_uniform_(self.a2.data, gain=gain)

        if self.bias_s is not None:
            self.bias_s.data.fill_(0)
            self.bias_t.data.fill_(0)

    def forward(self, hs, ht, A_opt):
        """Forward pass.

        Parameters
        ----------
        hs : torch.tensor, shape=[n_source_cell, source_in_features]
            Input features for source cells.
        ht : torch.tensor, shape=[n_target_cell, target_in_features]
            Input features for target cells.
        A_opt : torch.tensor, shape=[n_target_cell, n_source_cell].
            Typically this can be a boundary matrix
            when the operator is asymmetrical,
            the adjacency, laplacian matrix when
            the operator is symmetric.
        * If A_opt is symmetric:
            One should think about the this operator
            as a function that updates the signal
            or features on eeah source cell.
            Here source and target cells are the same.
            In other words, the source and the target are
            idenitical. This is similar to the graph attention
            case except here that we can have attention
            between edges, faces and higher dimensional cells/simplices.
        * If A_opt is asymmetric:
            One should think about this operator
            as a function that moves features
            hs (source) to feature ht ( target ).

        Returns
        -------
        * If A_opt is symmetric:
            h_prime_st : torch.tensor, shape=[n_source_cell, n_target_out_features],
                Here, num n_source_cell=n_target_cell
            h_prime_ts. None
        * If A_opt is asymmetric:
            h_prime_st : torch.tensor, shape=[n_source_cell, n_target_out_features]
            h_prime_ts. torch.tensor, shape=[n_target_cell, n_source_out_features]
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
                "[1, n_in_simplices, n_features],"
                "the tensor is being reshaped to "
                "[n_in_simplices, n_features].",
                stacklevel=2,
            )

            hs = hs.squeeze(0)

        if ht is not None:
            if len(ht.shape) == 3 and ht.shape[0] != 1:
                raise Exception(" batch multiplication is not supported.")
            if len(ht.shape) == 3 and ht.shape[0] == 1:
                warn(
                    "The first tensor has shape"
                    "[1, n_in_simplices, n_features],"
                    " the tensor is being reshaped to "
                    "[n_in_simplices, n_features].",
                    stacklevel=2,
                )

                ht = ht.squeeze(0)

        if hs.shape[0] != A_opt.shape[1]:
            raise ValueError(
                " n_source_cell in the second argument tensor hs  must match A_opt.shape[1]."
            )

        if ht is not None:
            if ht.shape[0] != A_opt.shape[0]:
                # TODO write better error
                raise ValueError(
                    "n_source_cell the first argument "
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

            if self.concatenate:
                return F.elu(h_prime_st), F.elu(h_prime_ts)
            return h_prime_st, h_prime_ts

        else:
            if self.concatenate:
                return F.elu(h_prime_st), None
            return h_prime_st, None

    def _prepare_higher_order_attentional_input(self, Whs, Wht):
        """Prepare the input to the HOAN.

        The purpose is to compute attentions simplex/cell s <-> simplex/cell t

        Rational:
        * If A_opt is symmetric:
            This case is not different from the graph case,
            the attention matrix here is symmetric and its
            size is identical the size of the input
            cochain linear operator.
        * If A_opt is asymmetric:
            When considering higher order attention
            between cells of different dimensions s and t,
            one needs signals on both the source and the
            target cells/simplicies.
            The attention matrix, between a cell with s and another
            cell t is determined by a coeff e[s,t] which represents
            the attention that is given to the cell t by the cell s.
            To consider the attenion e[t,s] one needs to look at
            the transpose relationship between the cells
            the sources and the target

        Parameters
        ----------
        Whs : torch.tensor, shape=[n_in_cell , n_source_out_features]
            Weight matrix.
        Wht : torch.tensor, shape=[n_out_cell , n_target_out_features]
            Weight matrix.

        Returns
        -------
        e_st : torch.tensor, shape=[n_target_cell , n_source_cell]
            Represents the attention between s -> t.
        e_ts : torch.tensor, shape=[n_source_cell , n_target_cell]
            Represents the attention between t -> s.
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
    """Class for a Higher Order Attention Layer (HOAN).

    This is a sparse implementation of a HOAN.

    Mathematically, a higher order attention layer can be defined
    in one of the following two ways, depending on whether the
    neighborhood matrix A is symmetric or not.

    * If A is symmetric:
        phi( A . att * X * W )
        where:
            - A is a symmetric cochain operator A:C^i -> C^i.
            The element of this matrix are assumed
            to be from the set {0,1}.
            - att is an attention layer that has the
            same shape as the operator A.
            - X is the input feature vector that belongs
            to the cochain space C^i.
            - W is a trainable parameter matrix.
            - phi is a non-linearity.
            - operation * is a matrix multiplication.
            - operation . is an element-wise multiplication.
    * If A is asymmetric:
            phi( A . att_s2t * X_s * W_s2t )
            phi( A.T . att_t2s * X_t * W_t2s )
        where:
            - A is a symmetric cochain operator A:C^s -> C^t
            (Dense rep. of A is of shape
            [n_target_cell, n_source_cell] ).
            The element of this matrix are assumed to
            be from the set {0,1}.
            - att_s2t is an attention matrix that has
            the same shape as the operator A,
            [n_target_cell, n_source_cell].
            - att_t2s is an attention matrix that has
            the same shape as the transpose of operator A,
            [n_source_cell,n_target_cell].
            - X_s is the input feature vector that
            belongs to the cochain space C^s.
            - X_t is the input feature vector that
            belongs to the cochain space C^t.
            - W_s2t, and W_t2s are trainable parameter matrices.

    Parameters
    ----------
    source_in_features : int
        Number of input features for the source cells.
    target_in_features : int
        Number of input features for the target cells.
    source_out_features : int
        Number of output features for the source cells.
    target_out_features : int
        Number of output features for the target cells.
    dropout : float, optional
        Dropout rate, by default 0.1.
    alpha : float, optional
        LeakyReLU angle of the negative slope, by default 0.1.
    concatenate : bool, optional
        Whether to concatenate the input features to the output ones,
        by default True.
    bias : bool, optional
        Whether to add a bias term, by default True.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        dropout=0.1,
        alpha=0.1,
        concatenate=True,
        bias=True,
    ):
        super(SparseHigherOrderAttentionLayer, self).__init__(
            source_in_features,
            target_in_features,
            source_out_features,
            target_out_features,
            dropout,
            alpha,
            concatenate,
            bias,
        )
        self.reset_parameters()

    def reset_parameters(self):
        """Reset parameters."""
        super().reset_parameters()

    def forward(self, hs, ht, operator_list, operator_symmetry=False):
        """Forward pass.

        Parameters
        ----------
        hs : torch.tensor, shape=[n_source_cell, source_in_features]
            Input features of the source cells.
        ht : torch.tensor, shape=[n_target_cell, target_in_features]
            Input features of the target cells.
        operator_list :
            This represents the cochain map C^s -> C^t.
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
        operator_symmetry: bool
            Indicates if the A is symmetric or not.

        Returns
        -------
        * If A is symmetric (operator_symmetry is True):
        h_prime_st : torch.tensor, shape=[n_source_cell, n_target_out_features]
                Here, num n_source_cell=n_target_cell.
        h_prime_ts : None.
        * If A is asymmetric (operator_symmetry is False):
        h_prime_st : torch.tensor, shape=[n_source_cell, n_target_out_features]
        h_prime_ts : torch.tensor, shape=[n_target_cell, n_source_out_features]
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
            if self.concatenate:
                return F.elu(h_prime_st), None
            else:
                return h_prime_st, None
        else:
            if self.concatenate:
                return F.elu(h_prime_st), F.elu(h_prime_ts)
            else:
                return h_prime_st, h_prime_ts
