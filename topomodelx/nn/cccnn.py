# --------------------------------------------------------
# Base classes for general conv operator executed on a toplogical space
# modeled as a combinatorial/cellular(CW)/simplicial/cubical/polyhedral complex.
#
# --------------------------------------------------------
__all__ = [
    "_LTN",
    "LTN",
    "BatchLTN",
    "BatchMergeOper",
    "BatchMultiMergeOper",
    "BatchMultiSplitOper",
    "BatchSplitOper",
    "MergeOper",
    "MultiMergeOper",
    "MultiSplitOper",
    "SplitOper",
    "_MergeOper",
    "_MultiMergeOper",
    "_MultiSplitOper",
    "_SplitOper",
]

from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from topomodelx.nn.linear import Linear
from topomodelx.util import batch_mm
from torch import Tensor
from torch.nn.parameter import Parameter

r"""
This class implements message passing functions on regular cell complexes
 (as well as simplicial complexes).
    Given a complex X (simplicial or cellular),
    the general form conv operator has the form:
    .. math::
    X^{t}_{i+1}= M(A_1,...,A_n,X^{s_1}_{i},...,X^{s_k}_{i} ),
The implementation given here is given in terms of
three main functions (LTN, Merge, Split).
These functions can be used to build an arbitrary network on the complex.
Specifically, this calss implements the following  :
    (1) LTN, implements the simpliest form of the above
        conv operator :math:`X^{t}_{i+1}= M(G,X^{s}_{i} ) =G X^s_{i} W`
        here :math:`G:C^s(X)->C^t(X)' is a cochain map with
        :math:`C^i(X)` being the linear space of all cochains of
        dimension :math:`i` defined on X.
        While message passing on complexes can be implemented
        in many ways, other than the convolutional way implemented
        in this class, we choose here the conv method to make it
        easier for developers to expand upon
        the space of applications of cell nets.
    (2) Merge operator, given two cochain
        :math:`x_1,x_2` in :math:`C^{i_1}(X),C^{i_2}(X)`,
        the merge operator merge x1 and x2 by sending them to a
        common cochains spaces :math:`C^j(X)` using two linear maps
        :math:`G_1:C^{i_1}(X)->C^j(X)` and  :math:`G_2:C^{i_2}(X)->C^j(X)`.
    (3) Split operator, given a cochain x in :math:`C^i(X)`,the split
        operator sends x to two differnt cochains spaces :math:`C^j(X)` and
        C^k(X) using two linear maps :math:`G1:C^i(X)->C^j(X)`
        and  :math:`G_2:C^i(X)->C^k(X)`.
    (4) MultiMerge operator, generlize the above merge operator to n merges.
    (5) MutliSplit operator, generlize the above split operator to n splits.
    (6) Batch versions of all the above operators are also supported.
"""


class _LTN(nn.Module):
    r"""
    LTN, implements the simpliest form of the above
    conv operator on that can be defined in the context of
    complexes that appear in the algebraic
    topology (simplicial, cellular, polyhedral, etc).
    Precisly, LTN takes the form:
    .. math::
    X^{t}_{i+1}= M(G,X^{s}_{i}) =G * X^s_{i} * W  ----[1],
    where :math:`G:C^s(X)->C^t(X)`, where :math:`C^i(X)` is the linear space
    of all cochains of dimension i living on X.
    :math:`X^s_{i}` is cochain in  :math:`C^s(X)`
    W is a trainable parameter.
    [1] This is essentially an implementation of equation 2.3
       given in https://openreview.net/pdf?id=6Tq18ySFpGU
    See also [2,3].
    Introduction:
    -------------
        An A_operator :math:`C^{in}(X)->C^{out}(X)`
        is a matrix that moves a cochain x
        that reside on all simplices/cells on X of a specific dimension
        (here it is the "in" dimension) to a signal that lives on
        on simplicies of dimension "out". Concretly, A_operator is a
        cochain map that operators that sends a signal in :math:`C^{in}(X)`
        to a signal in :math:`C^{in}(X)`.
        Given the operator A_opt, the LTN operator induced by it
        is also a map that operates between the same cochain spaces.
         Assuming x is of shape [num_in_cell, num_features_in ]
        then typically A_operator is a
        (co)boundary matrix/ k-Hodge Laplacian/k-(co)adjacency/
        matrix of shape [num_out_cell,num_in_cell ].
        Args:
            in_ft (int): dimension of input features.
            out_ft (int): positive int, dimension of out features.
            dropout (float,optional): default is 0.0.
            bias (bool,optional): default is True.
            init_scheme (str,optional): the initializer for the
                weight, default is xavier, other options : debug.
    Ref:
    ----
    [2] Roddenberry, T. Mitchell, Nicholas Glaze, and Santiago Segarra.
    "Principled simplicial neural networks for trajectory prediction."
    International Conference on Machine Learning. PMLR, 2021.
    [3] Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
    "Signal processing on cell complexes."
    arXiv preprint arXiv:2110.05614 (2021)."""

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
        batch_cochain=True,
    ):
        super(_LTN, self).__init__()

        self.in_ft = in_ft
        self.out_ft = out_ft
        self.init_scheme = init_scheme
        self.batch_cochain = batch_cochain
        self.dropout = dropout

        # create the paramers of the model
        if self.batch_cochain:
            # extra channel added in the begining for batch multiplication
            self.weight = Parameter(torch.Tensor(1, in_ft, out_ft))
            if bias:
                # extra channel added in the begining for batch multiplication
                self.bias = Parameter(torch.Tensor(1, out_ft))
            else:
                self.register_parameter("bias", None)

        else:
            self.weight = Parameter(torch.Tensor(in_ft, out_ft))
            if bias:
                self.bias = Parameter(torch.Tensor(out_ft))
            else:
                self.register_parameter("bias", None)
        self.reset_parameters()

    def reset_parameters(self, gain=1.414):

        if self.bias is not None:

            nn.init.zeros_(self.bias)

        if self.init_scheme == "xavier_uniform":

            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.init_scheme == "xavier_normal":

            nn.init.xavier_normal_(self.weight, gain=gain)

        elif self.init_scheme == "debug":
            # typically utilized during debugging.
            warn(
                "This init scheme is typically used during debugging, ",
                "in this case it is also typical to set dropout to zero.",
            )

            if self.batch_cochain:
                nn.init.eye_(self.weight.squeeze(0))

            else:
                nn.init.eye_(self.weight)

        elif self.init_scheme == "uniform":

            stdv = 1.0 / torch.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.init_scheme}' is not supported"
            )

    def forward(self, x: Tensor, A_operator: Tensor) -> Tensor:
        r"""
        Args:
        -------
            if batch_cochain is True:
                x : cellular/simplicial features - Tensor with cell
                    features of shape
                    [batch_size, num_in_cell, num_features_in]
            if batch_cochain is False:
                x : cellular/simplicial features - Tensor with cell
                    features of shape [um_in_cell, num_features_in]
            A_operator : a cochain matrix that represents a
            cochain map C^i->C^j . Entry A_operator[i,j]=1 means there
            is a message from cell/simplex i to cell/simplex j .
        output:
        -------
            pytorch tensor x:
                 Shape : [batch_size,num_cells_out,num_features_out ]
                     if batch_cochain=True
                 Shape : [num_cells_out,num_features_out ]
                     if batch_cochain=False
        """

        if A_operator in [None, "Id"]:
            if not isinstance(x, torch.Tensor):
                raise TypeError("input x must be a torch tensor")
            if self.in_ft != self.out_ft:
                raise ValueError(
                    "The input operator is None or 'Id' acts",
                    " as an identity operator,",
                    "the in_ft must be the same as out_ft ",
                    "in the model constructor.",
                )

            if self.batch_cochain:
                if len(x.shape) == 2:
                    warn(
                        "Input tensor has shape of the form"
                        + "[ # of input cells , # of feature channels ]"
                        + "with no batch channel, the input tensor"
                        + " shape will be changed to"
                        + "[ 1,# of input cells , # of feature channels ]."
                        + "This shape change also reflects"
                        + " on the shape of the output"
                        + "tensor and one batch channel "
                        + "will be added at the top of "
                        + "the output tensor.",
                        stacklevel=2,
                    )

                    x = x.unsqueeze(0)
                    assert len(x.shape) == 3

                if self.in_ft != x.shape[-1]:
                    raise ValueError(
                        "The input operator is None acts as "
                        "an identity operator, the in_ft must be"
                        " the same as number of features in the input cochain"
                    )

                return x

            else:
                if self.in_ft != x.shape[-1]:
                    raise ValueError(
                        "The input operator is None acts as an identity "
                        + "operator, the in_ft must be the same as number"
                        + " of features in the input cochain"
                    )

                return x

        if self.batch_cochain:
            if not isinstance(A_operator, torch.Tensor):
                raise TypeError(
                    "Input operator must be torch tensor,"
                    + " instead got an input of type ",
                    type(A_operator),
                )
            if not isinstance(x, torch.Tensor):
                raise TypeError(
                    "Input cochain must be torch tensor. "
                    "Instead got an input of type ",
                    type(x),
                )

            if len(x.shape) == 3:  # assuming batch input
                if x.shape[1] != A_operator.shape[-1]:
                    raise ValueError(
                        "Mumber of source cells/simplicies must match "
                        + "number of elements in input vector. "
                        + "number of elements in the input vector is "
                        + f"{x.shape[1]} and number of source cells is {A_operator.shape[-1]}."
                    )
            elif (
                len(x.shape) == 2
            ):  # Assuming single input, batchsize=1 and no batch channel is included
                if x.shape[0] != A_operator.shape[-1]:
                    raise ValueError(
                        "Number of source cells/simplicies must match number of elements in input tensor."
                        + f"Number of elements in the input vector is {x.shape[0]} "
                        + f"and number of source cells is {A_operator.shape[-1]}."
                    )
                    raise

            if len(x.shape) == 2:
                warn(
                    "Input tensor has shape of the form"
                    + " [ number of input cells , number of feature channels ]"
                    + " with no batch channel, the input tensor shape will be"
                    + " changed to"
                    + " [1,# of input cells, # of feature channels]."
                    + " This shape change also reflects on the shape"
                    + " of the output tensor "
                    + " and one batch channel will be added"
                    + " at the top of the output tensor.",
                    stacklevel=2,
                )

                x = x.unsqueeze(0)
                assert len(x.shape) == 3

            x = F.dropout(x, self.dropout, self.training)

            if x.is_sparse:
                x = torch.bmm(x.to_dense(), self.weight)
            else:
                x = x.matmul(self.weight)

            if A_operator.is_sparse:
                x = batch_mm(A_operator, x)
                # x = torch.spmm(A_operator, x) # TODO: spmm does not work with stride multiplication
            else:
                x = A_operator.matmul(x)

            if self.bias is not None:
                if x.is_sparse:
                    x = x.to_dense()

                x = self.bias + x

            return x

        else:  # nonbatch multiplication.
            if not isinstance(x, Tensor):
                raise TypeError(
                    f"Input cochain must be torch tensor. Instead got an input of type {type(x)}."
                )

            if (
                len(x.shape) == 2
            ):  # assuming single input, batchsize=1 and no batch channel is included
                if x.shape[0] != A_operator.shape[-1]:
                    raise ValueError(
                        "Number of source cells/simplicies must match number of elements in input tensor."
                        + f"Number of elements in the input vector is {x.shape[0]} "
                        + f"and number of source cells is {A_operator.shape[-1]}."
                    )
            else:
                Exception(
                    "number of channels in the input cochain tensor must be 2, got number of channels"
                    + str(len(x.shape))
                )
            x = F.dropout(x, self.dropout, self.training)

            support = torch.mm(x, self.weight)

            output = torch.spmm(A_operator, support)

            if self.bias is not None:
                if output.is_sparse:
                    output = output.to_dense()

                support = self.bias + output

            return output

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_ft)
            + " -> "
            + str(self.out_ft)
            + ")"
        )


class LTN(_LTN):
    r"""
     LTN, implements the simpliest form of the above
     conv operator on that can be defined in the context of
     complexes that appear in the algebraic
     topology (simplicial, cellular, polyhedral, etc).
     Precisly, LTN takes the form:
     .. math::
     X^{t}_{i+1}= M(G,X^{s}_{i}) =G * X^s_{i} * W  ----[1],
     where :math:`G:C^s(X)->C^t(X)`, where :math:`C^i(X)` is the linear space
     of all cochains of dimension i living on X.
     :math:`X^s_{i}` is cochain in  :math:`C^s(X)`
     W is a trainable parameter
    [1] This is essentially an implementation of equation 2.3
        given in https://openreview.net/pdf?id=6Tq18ySFpGU. See also [2,3].
     Introduction:
     -------------
         An A_operator :math:`C^{in}(X)->C^{out}(X)`
         is a matrix that moves a cochain x
         that reside on all simplices/cells on X of a specific dimension
         (here it is the "in" dimension) to a signal that lives on
         on simplicies of dimension "out". Concretly, A_operator is a
         cochain map that operators that sends a signal in :math:`C^{in}(X)`
         to a signal in :math:`C^{in}(X)`.
         Given the operator A_opt, the LTN operator induced by it
         is also a map that operates between the same cochain spaces.
          Assuming x is of shape [num_in_cell, num_features_in ]
         then typically A_operator is a
         (co)boundary matrix/ k-Laplacian/k-adjacency
         matrix of shape [num_out_cell,num_in_cell ].
     Ref:
     ----
     [2] Roddenberry, T. Mitchell, Nicholas Glaze, and Santiago Segarra.
     "Principled simplicial neural networks for trajectory prediction."
     International Conference on Machine Learning. PMLR, 2021.
     [3] Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
     "Signal processing on cell complexes."
     arXiv preprint arXiv:2110.05614 (2021)."""

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(LTN, self).__init__(in_ft, out_ft, dropout, bias, init_scheme, False)
        """
        Args:
            in_ft: positive int, dimension of input features
            out_ft: positive int, dimension of out features
            dropout: optional, default is 0.1
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug.
        """

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()


class BatchLTN(_LTN):
    r"""
     BatchLTN, a batch versio of LTN and it
     implements the simpliest form of the above
     conv operator on that can be defined in the context of
     complexes that appear in the algebraic
     topology (simplicial, cellular, polyhedral, etc).
     Precisly, LTN takes the form:
     .. math::
     X^{t}_{i+1}= M(G,X^{s}_{i}) =G * X^s_{i} * W  ----[1],
     where :math:`G:C^s(X)->C^t(X)`, where :math:`C^i(X)` is the linear space
     of all cochains of dimension i living on X.
     :math:`X^s_{i}` is cochain in  :math:`C^s(X)`
     W is a trainable parameter
    [1] This is essentially an implementation of equation 2.3
        given in https://openreview.net/pdf?id=6Tq18ySFpGU
        See also [2,3]
     Introduction:
     -------------
         An A_operator :math:`C^{in}(X)->C^{out}(X)`
         is a matrix that moves a cochain x
         that reside on all simplices/cells on X of a specific dimension
         (here it is the "in" dimension) to a signal that lives on
         on simplicies of dimension "out". Concretly, A_operator is a
         cochain map that operators that sends a signal in :math:`C^{in}(X)`
         to a signal in :math:`C^{in}(X)`.
         Given the operator A_opt, the LTN operator induced by it
         is also a map that operates between the same cochain spaces.
          Assuming x is of shape [num_in_cell, num_features_in ]
         then typically A_operator is a
         (co)boundary matrix/ k-Laplacian/k-adjacency
         matrix of shape [num_out_cell,num_in_cell ].
     Ref:
     ----
     [2] Roddenberry, T. Mitchell, Nicholas Glaze, and Santiago Segarra.
     "Principled simplicial neural networks for trajectory prediction."
     International Conference on Machine Learning. PMLR, 2021.
     [3] Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
     "Signal processing on cell complexes."
     arXiv preprint arXiv:2110.05614 (2021).
    """

    def __init__(
        self,
        in_ft: int,
        out_ft: int,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):
        super(BatchLTN, self).__init__(in_ft, out_ft, dropout, bias, init_scheme, True)
        """
        Args:
        ------
            in_ft: positive int, dimension of input features
            out_ft: positive int, dimension of out features
            dropout: optional, default is 0.0
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

        self.reset_parameters()

    def reset_parameters(self):
        super().reset_parameters()


class _MergeOper(nn.Module):
    r"""
    Description
    -----------
     Given two cochain x1,x2 in C^i1(X),C^i2(X), the merge operator
     merge x1 and x2 by sending them to a common cochains spaces C^j(X)
     using two linear maps G1:C^i1(X)->C^j(X) and  G2:C^i2(X)->C^j(X).
    Args:
    ----------
        in_ch_1 : number of features in the first input channel
        in_ch_2 : number of features in the second input channel
        target_ch :  number of features in the output channel
        in_ft: positive int, dimension of input features
        out_ft: positive int, dimension of out features
        dropout: optional, default is 0.1
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_1: int,
        in_ch_2: int,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=False,
        merge_type="sum",
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
        batch_cochain=False,
    ):

        super(_MergeOper, self).__init__()

        self.merge_type = merge_type
        self.in_ch_1 = in_ch_1
        self.in_ch_2 = in_ch_2
        self.target_ch = target_ch
        self.batch_cochain = batch_cochain
        self.shared_parameters = shared_parameters
        self.initial_linear_shift = initial_linear_shift

        if self.shared_parameters:
            if self.initial_linear_shift:
                self.linear1 = Linear(in_ch_1, target_ch)
                self.linear2 = Linear(in_ch_2, target_ch)

                if self.batch_cochain:
                    self.LN3 = BatchLTN(
                        target_ch,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                else:
                    self.LN3 = LTN(
                        target_ch,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
            else:
                if self.in_ch_1 != self.in_ch_2:
                    raise ValueError(
                        "input channels must be the same when 'initial_linear_shift' is False"
                    )

                if self.batch_cochain:
                    self.LN3 = BatchLTN(
                        self.in_ch_1,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                else:
                    self.LN3 = LTN(
                        self.in_ch_1,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )

        else:
            if self.batch_cochain:

                self.LN1 = BatchLTN(
                    in_ch_1,
                    target_ch,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
                self.LN2 = BatchLTN(
                    in_ch_2,
                    target_ch,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
            else:
                self.LN1 = LTN(
                    in_ch_1,
                    target_ch,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
                self.LN2 = LTN(
                    in_ch_2,
                    target_ch,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )

    def forward(self, x1: Tensor, x2: Tensor, G1: Tensor, G2: Tensor) -> Tensor:
        r"""
        Args:
        ------
            x1,x2 : torch tensors representing cell/simplicial features -
            x1 is of shape [batch_size, num_source_cells_1, in_ch_1]
            x2 is of shape [batch_size, num_source_cells_2, in_ch_2]
            G1 - torch tensor representing a cochain matrix that
            represents a cochain map C^i1->C^j . Entry A_operator[i,j]=1
            means there is a message from cell/simplex i to cell j .
                         Shape: [ num_target_cells, num_source_cells_1]
            G2 - torch tensor representing a cochain matrix that represents
            a cochain map C^i1->C^j . Entry A_operator[i,j]=1 means there is
            a message from cell/simplex i to cell j.
                         Shape: [ num_target_cells, num_source_cells_2]
        output:
        ------
            pytorch tensor x:
                 Shape : [batch_size,num_cells_out,num_features_out  ]
        """
        if x1 is None and x2 is not None:
            if G1 is not None:
                raise ValueError(
                    "when the first input tensor is None"
                    "the first merge operator must be also None or Id"
                )
                return self.LN1(x2, G2)
        elif x1 is not None and x2 is None:
            if G2 is not None:
                raise ValueError(
                    "when the first input tensor is None"
                    "the first merge operator must be also None or Id"
                )
            return self.LN2(x1, G1)
        elif x1 is not None and x2 is not None:
            if self.batch_cochain:
                if len(x1.shape) != 3 or len(x2.shape) != 3:
                    raise ValueError(
                        "with batch merge, both input"
                        + " tensor must be three dimensionals,"
                        + f"got tensors of shape {x1.shape}"
                        + f"and {x2.shape}"
                    )
                if x1.shape[0] != x2.shape[0]:
                    raise ValueError(
                        "input tensors x1 and x2 must " "have the same batch size"
                    )
            if G1 is None and G2 is None:
                if not (self.in_ch_1 == self.in_ch_2 == self.target_ch):
                    raise ValueError(
                        "when the types of the operators G1 and G2 are None, "
                        "in_ch_1, in_ch_2 and target_ch must be equal"
                    )
                if x1.shape[-1] != x2.shape[-1]:
                    raise ValueError(
                        "input cochain tensors x1 and x2",
                        " must have the same dimensions",
                    )
            elif G1 is None and G2 is not None:
                if self.in_ch_1 != self.target_ch:
                    raise ValueError(
                        " when  the operator G1 is None,"
                        " target_ch_1, and num_features_in must be equal"
                    )
            elif G1 is not None and G2 is None:
                if self.in_ch_2 != self.target_ch:
                    raise ValueError(
                        " when the operator G2 is None, "
                        "target_ch_2, and num_features_in must be equal"
                    )
            else:
                if G1.shape[0] != G2.shape[0]:
                    raise ValueError(
                        "Input operators G1 and G2" " must have the same target "
                    )
            if self.shared_parameters:
                if self.initial_linear_shift:
                    x1 = self.linear1(x1)
                    x2 = self.linear2(x2)
                if self.merge_type == "sum":
                    x = self.LN3(x1, G1) + self.LN3(
                        x2, G2
                    )  # merge the two vectors--to do add more merging operations
                elif self.merge_type == "average":
                    x = torch.mean(
                        self.LN3(x1, G1), self.LN3(x2, G2)
                    )  # merge the two vectors--to do add more merging operations
                else:
                    raise RuntimeError(
                        f" merge_type must be either sum or average "
                        f"'{self.merge_type}' is not supported"
                    )
                return x
            else:
                if self.merge_type == "sum":
                    x = self.LN1(x1, G1) + self.LN2(
                        x2, G2
                    )  # merge the two vectors--to do add more merging operations
                elif self.merge_type == "average":
                    x = torch.mean(
                        self.LN1(x1, G1), self.LN2(x2, G2)
                    )  # merge the two vectors--to do add more merging operations
                elif self.merge_type == "concat":
                    x = torch.stack([self.LN1(x1, G1), self.LN2(x2, G2)])
                else:
                    raise RuntimeError(
                        f" merge_type must be either sum or average "
                        f"'{self.merge_type}' is not supported"
                    )
                return x
        else:
            raise RuntimeError(
                "invalid merge request, both input tenors cannot be None"
            )

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_ch_1)
            + " ,"
            + str(self.in_ch_2)
            + " -> "
            + str(self.target_ch)
            + ")"
        )


class MergeOper(_MergeOper):
    r"""
    Description
    -----------
     Given two cochain x1,x2 in C^i1(X),C^i2(X), the merge operator
     merge x1 and x2 by sending them to a common cochains spaces C^j(X)
     using two linear maps G1:C^i1(X)->C^j(X) and  G2:C^i2(X)->C^j(X).
     Args
     -----
        in_ch_1 : number of features in the first input channel
        in_ch_2 : number of features in the second input channel
        target_ch :  number of features in the output channel
        in_ft: positive int, dimension of input features
        out_ft: positive int, dimension of out features
        dropout: optional, default is 0.0
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_1: int,
        in_ch_2: int,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=True,
        merge_type="sum",
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(MergeOper, self).__init__(
            in_ch_1,
            in_ch_2,
            target_ch,
            shared_parameters,
            initial_linear_shift,
            merge_type,
            dropout,
            bias,
            init_scheme,
            False,
        )


class BatchMergeOper(_MergeOper):
    r"""
    Description
    -----------
     Given two cochain x1,x2 in :math:`C^{i_1}(X),C^{i_2}(X)`,
     the merge operator merge :math:`x_1`
     and :math:`x_2` by sending them to a common
     cochain space :math:`C^j(X)` using two
     linear maps :math:`G_1:C^{i_1}(X)\to C^j(X)`
     and :math:`G_2:C^{i_2}(X)\to C^j(X)`.
    Args
    -----
        in_ch_1 : number of features in the first input channel
        in_ch_2 : number of features in the second input channel
        target_ch :  number of features in the output channel
        in_ft: positive int, dimension of input features
        out_ft: positive int, dimension of out features
        dropout: optional, default is 0.1
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_1: int,
        in_ch_2: int,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=True,
        merge_type="sum",
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(BatchMergeOper, self).__init__(
            in_ch_1,
            in_ch_2,
            target_ch,
            shared_parameters,
            initial_linear_shift,
            merge_type,
            dropout,
            bias,
            init_scheme,
            batch_cochain=True,
        )


class _SplitOper(nn.Module):
    r"""
    Description
    -----------
    Given a cochain x in :math:`C^i(X)`, the split operator sends x
    to two differnt cochains spaces :math:`C^j(X)` and :math:`C^k(X)`
    using two linear maps
    :math:`G1:C^i(X)\to C^j(X)` and :math:`G2:C^i(X)\to C^k(X)`.
    Args
    -----
        num_features_in : number of features in the input channel
        target_ch_1 : number of features in the first output channel
        target_ch_2 : number of features in the second output channel
        in_ft: positive int, dimension of input features
        out_ft: positive int, dimension of out features
        dropout: optional, default is 0.1
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_ch_1: int,
        target_ch_2: int,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
        batch_cochain=True,
    ):

        super(_SplitOper, self).__init__()
        self.num_features_in = num_features_in
        self.target_ch_1 = target_ch_1
        self.target_ch_2 = target_ch_2
        self.batch_cochain = batch_cochain
        self.shared_parameters = shared_parameters

        if self.shared_parameters:
            if self.target_ch_1 != self.target_ch_2:
                raise ValueError(
                    "The output channels must be equal when shared_parameters is True"
                )

            if self.batch_cochain:
                self.LN = BatchLTN(
                    num_features_in,
                    target_ch_1,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
            else:
                self.LN = LTN(
                    num_features_in,
                    target_ch_1,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
        else:
            if self.batch_cochain:

                self.LN1 = BatchLTN(
                    num_features_in,
                    target_ch_1,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
                self.LN2 = BatchLTN(
                    num_features_in,
                    target_ch_2,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
            else:
                self.LN1 = LTN(
                    num_features_in,
                    target_ch_1,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
                self.LN2 = LTN(
                    num_features_in,
                    target_ch_2,
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )

    def forward(self, x: Tensor, G1: Tensor, G2: Tensor) -> Tensor:
        r"""
        Args:
        ------
            x: torch tensor representing cell/simplicial features -
                x is of shape [batch_size, num_in_cell, num_features_in]
            G1 - torch tensor representing a cochain matrix
                that represents a cochain map C^i1->C^j.
                Entry A_operator[i,j]=1 means there is a message from
                cell/simplex i to cell/simplex j .
                         Shape: [ num_target_cells_1, num_source_cells]
            G2 - torch tensor representing a cochain matrix that
                represents a cochain map C^i1->C^j.
                Entry A_operator[i,j]=1 means there is a message
                from cell/simplex i to cell/simplex j .
                         Shape: [ num_target_cells_2,num_source_cells]
        output:
        -------
            a pytorch tensors x1:
                 x1 Shape : [batch_size,num_target_cells_1,target_ch_1  ]
            a pytorch tensors x2:
                 x2 Shape : [batch_size,num_target_cells_2,target_ch_2  ]
        """
        if G1 is None and G2 is None:
            if not (self.target_ch_1 == self.target_ch_2 == self.num_features_in):
                raise ValueError(
                    " when the types of the operators",
                    " G1 and G2 None, target_ch_1,",
                    " target_ch_2 and num_features_in must be equal",
                )

        elif G1 is None and G2 is not None:
            if self.target_ch_1 != self.num_features_in:
                raise ValueError(
                    "when the types of the operators",
                    "G1 is None then target_ch_1",
                    "and num_features_in must be equal,",
                    f"got target_ch_1={self.target_ch_1}",
                    f"and num_features_in={self.num_features_in}",
                )

        elif G1 is not None and G2 is None:
            if self.target_ch_2 != self.num_features_in:
                raise ValueError(
                    "when the types of the operators G2 is None,"
                    " target_ch_2, and num_features_in must be equal."
                )
        else:
            if G1.shape[-1] != G2.shape[-1]:  # Input domains must be the same
                raise ValueError(
                    f"Operators must have the same domain, input domains are {G1.shape[-1]} and {G2.shape[-1]}."
                )

        if self.shared_parameters:
            x1 = self.LN(x, G1)
            x2 = self.LN(x, G2)
        else:
            x1 = self.LN1(x, G1)
            x2 = self.LN2(x, G2)

        return x1, x2

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.num_features_in)
            + " -> "
            + str(self.target_ch_1)
            + " ,"
            + str(self.in_ch_2)
            + ")"
        )


class SplitOper(_SplitOper):
    r"""
    Description
    -----------
     Given a cochain x in C^i(X)
     the split operator splits sending
     them to two cochains spaces C^{j1}(X),  C^{j2}(X)
     using two linear maps G_1:C^{i}(X)->C^{j_2}(X) and
     G_2:C^{i}(X)->C^{j_2}(X).
    Args
    ----
        num_features_in : number of features in the input channel
        target_ch_1 : number of features in the first output channel
        target_ch_2 : number of features in the second output channel
        dropout: optional, default is 0.0
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_ch_1: int,
        target_ch_2: int,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(SplitOper, self).__init__(
            num_features_in,
            target_ch_1,
            target_ch_2,
            shared_parameters,
            dropout,
            bias,
            init_scheme,
            False,
        )


class BatchSplitOper(_SplitOper):
    r"""
    Description
    -----------
     Given a cochain x in C^i(X)
     the split operator splits sending
     them to two cochains spaces C^{j1}(X),  C^{j2}(X)
     using two linear maps G_1:C^{i}(X)->C^{j_2}(X) and
     G_2:C^{i}(X)->C^{j_2}(X).
    Args
    ----
        num_features_in : number of features in the input channel
        target_ch_1 : number of features in the first output channel
        target_ch_2 : number of features in the second output channel
        dropout: optional, default is 0.0
        bias: optional, default is True
        init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_ch_1: int,
        target_ch_2: int,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(BatchSplitOper, self).__init__(
            num_features_in,
            target_ch_1,
            target_ch_2,
            shared_parameters,
            dropout,
            bias,
            init_scheme,
            True,
        )


class _MultiMergeOper(nn.Module):
    """
    Description
    -----------
        Given n cochains :math:`x_1,...,x_n` in
        :math:`C^{i_1}(X),..,C^{i_n}(X)`,
        the merge operator merge :math:`x_{i_k}` by sending them
        to a common cochains space :math:`C^j(X)` using
        n linear maps :math:`G_{i_k}:C^{i_k}(X)->C^j(X) 1=<k<=n`.
        Remark: Essentially this class implements equation (8) given in
            Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
            "Signal processing on cell complexes."
            arXiv preprint arXiv:2110.05614 (2021).
        Args
        ----------
            in_ch_list: a list of positive integers,
                each in_ch_i is the number of features in the ith input channel
            target_ch :  number of features in the output channel
            in_ft: positive int, dimension of input features
            dropout: optional, default is 0.0
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_list: list,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=False,
        merge_type="sum",  # options: 'sum,max,min,average'
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
        batch_cochain=True,
    ):

        super(_MultiMergeOper, self).__init__()

        self.merge_type = merge_type
        self.in_ch_list = in_ch_list
        self.target_ch = target_ch
        self.batch_cochain = batch_cochain
        self.shared_parameters = shared_parameters
        self.initial_linear_shift = initial_linear_shift

        if self.shared_parameters:
            if self.initial_linear_shift:
                self.Linear_list = nn.ModuleList(
                    [
                        Linear(in_ch_list[i], target_ch)
                        for i in range(0, len(in_ch_list))
                    ]
                )

                if self.batch_cochain:
                    self.LN = BatchLTN(
                        target_ch,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                else:
                    self.LN = LTN(
                        target_ch,
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )

            else:
                if len(set([int(i) for i in in_ch_list])) != 1:
                    raise ValueError(
                        "input channels must be the same when 'initial_linear_shift' is False"
                    )

                if self.batch_cochain:
                    self.LN = BatchLTN(
                        in_ch_list[0],
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                else:
                    self.LN = LTN(
                        in_ch_list[0],
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
        if self.batch_cochain:

            self.LN_list = nn.ModuleList(
                [
                    BatchLTN(
                        in_ch_list[i],
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                    for i in range(0, len(in_ch_list))
                ]
            )
        else:
            self.LN_list = nn.ModuleList(
                [
                    LTN(
                        in_ch_list[i],
                        target_ch,
                        dropout=dropout,
                        bias=bias,
                        init_scheme=init_scheme,
                    )
                    for i in range(0, len(in_ch_list))
                ]
            )

    def forward(self, x_input_list: list, Gi_input_list: list) -> Tensor:
        r"""
        Args:
        ------
            x_input_list : list of torch tensors
                representing cell/simplicial features -
                xi is of shape [batch_size, num_source_cells_k, in_ch_k]
            Gi_input_list - list of torch tensor representing
                a cochain matrix that represents a cochain map :math:`G_k: C^{i_k}->C^j`.
                Shape: [ num_target_cells, num_source_cells_k]
        output:
        ------
            pytorch tensor x:
                 Shape : [batch_size,num_target_cells,target_ch  ]
        """

        """
        try:
            assert(G1.shape[0]==G2.shape[0]) # target1 of G1 must be == target2 of G2
        except AssertionError:
            print(" Input operators must have the same target ")
            raise
        if len(x1.shape)==3:
            try :
                assert(x1.shape[0]==x2.shape[0]) #  batch number must be the same for both inputs
            except AssertionError:
                print("input tensors x1 and x2 must have the same batch size")
                raise"""
        if len(x_input_list) != len(Gi_input_list):
            raise ValueError(
                "tenors list and operators list must have the same lengths."
            )
        # TODO, assertion on types of the input
        # TODO, assert the dimension of the input shape match with input tensors
        # TODO, fix dropout
        if set(x_input_list) is {None}:
            raise ValueError(
                "at least one tensor in the input tensor list must be not None to perform a valid merge."
            )

        # store indices tensors that are not None
        in_j = []
        for i in range(0, len(x_input_list)):
            if x_input_list[i] is None:
                if Gi_input_list[i] is not None:
                    raise ValueError(
                        "when the input tensor is None, ",
                        "the corresponding merge tensor must also be None",
                    )
            else:
                in_j.append(i)
        if self.shared_parameters:
            if self.initial_linear_shift:
                # only consider the ones that are not None
                pre_evals = [
                    self.Linear_list[in_j[i]](x_input_list[in_j[i]])
                    for i in range(0, len(in_j))
                ]
            else:
                pre_evals = [x_input_list[i] for i in in_j]
            lst = [
                self.LN(pre_evals[i], Gi_input_list[in_j[i]])
                for i in range(0, len(in_j))
            ]
        else:
            # only consider the ones that are not None
            lst = [
                self.LN_list[in_j[i]](x_input_list[in_j[i]], Gi_input_list[in_j[i]])
                for i in range(0, len(in_j))
            ]
        if len(in_j) == 1:
            return lst[0]
        if self.merge_type == "sum":
            x = torch.stack(lst).sum(axis=0)
        elif self.merge_type == "average":
            x = torch.stack(lst).mean(axis=0)
        elif self.merge_type == "max":
            x = torch.stack(lst).max(axis=0)
        elif self.merge_type == "min":
            x = torch.stack(lst).min(axis=0)
        elif self.merge_type == "concat":
            x = torch.stack(lst)
        else:
            raise Exception(
                "merge_type must be a string" " from [sum, average, min, max,concat]."
            )
        return x

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_ch_list)
            + ") -> "
            + str(self.target_ch)
        )


class MultiMergeOper(_MultiMergeOper):
    r"""
    Description
    -----------
        Given n cochains :math:`x_1,...,x_n` in
        :math:`C^{i_1}(X),..,C^{i_n}(X)`,
        the merge operator merge :math:`x_{i_k}` by sending them
        to a common cochains space :math:`C^j(X)` using
        n linear maps :math:`G_{i_k}:C^{i_k}(X)\to C^j(X) 1=<k<=n`.
        Remark: Essentially this class implements equation (8) given in
            Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
            "Signal processing on cell complexes."
            arXiv preprint arXiv:2110.05614 (2021).
        Args
        ----------
            in_ch_list: a list of positive integers,
                        each in_ch_i is the number of features in
                        the ith input channel.
            target_ch :  number of features in the output channel.
            in_ft: positive int, dimension of input features.
            dropout: optional, default is 0.0.
            bias: optional, default is True.
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_list: list,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=True,
        merge_type="sum",
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(MultiMergeOper, self).__init__(
            in_ch_list,
            target_ch,
            shared_parameters,
            initial_linear_shift,
            merge_type,
            dropout,
            bias,
            init_scheme,
            False,
        )


class BatchMultiMergeOper(_MultiMergeOper):
    r"""
    Description
    -----------
        Given n cochains :math:`x_1,...,x_n` in
        :math:`C^{i_1}(X),..,C^{i_n}(X)`,
        the merge operator merge :math:`x_{i_k}` by sending them
        to a common cochains space :math:`C^j(X)` using
        n linear maps :math:`G_{i_k}:C^{i_k}(X)->C^j(X) 1=<k<=n`.
        Remark: Essentially this class implements equation (8) given in
            Roddenberry, T. Mitchell, Michael T. Schaub, and Mustafa Hajij.
            "Signal processing on cell complexes."
            arXiv preprint arXiv:2110.05614 (2021).
        Args
        ----
            in_ch_list: a list of positive integers, each in_ch_i is the
                        number of features in the ith input channel
            target_ch :  number of features in the output channel
            in_ft: positive int, dimension of input features
            dropout: optional, default is 0.1
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        in_ch_list: list,
        target_ch: int,
        shared_parameters=False,
        initial_linear_shift=True,
        merge_type="sum",
        dropout=0.1,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(BatchMultiMergeOper, self).__init__(
            in_ch_list,
            target_ch,
            shared_parameters,
            initial_linear_shift,
            merge_type,
            dropout,
            bias,
            init_scheme,
            batch_cochain=True,
        )


class _MultiSplitOper(nn.Module):
    r"""
    Description
    -----------
        Given a cochain x in :math:`C^i(X)`, the split operator sends x to n
        differnt cochains spaces :math:`C^{j_k}(X)`
        using n linear maps
        :math:`G_1:C^i(X) \to C^{j_k}(X)` for :math`0<=k<n`.
        Args
        ----
            num_features_in : number of features in the input channel
            target_channels_list : list of number of output features channels
            dropout: optional, default is 0.1
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_channels_list: list,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
        batch_cochain=False,
    ):
        super(_MultiSplitOper, self).__init__()
        assert isinstance(target_channels_list, list)
        if len(target_channels_list) < 1:
            raise ValueError(
                " target channels must be larger than"
                " or equal to 1 in the split operator."
            )

        self.batch_cochain = batch_cochain
        self.num_features_in = num_features_in
        self.target_channels_list = target_channels_list
        self.shared_parameters = shared_parameters
        if self.shared_parameters:
            if len(set([int(i) for i in self.target_channels_list])) != 1:
                raise ValueError(
                    "The output channels must be equal when shared_parameters is True"
                )

            if self.batch_cochain:
                self.LN = BatchLTN(
                    num_features_in,
                    self.target_channels_list[0],
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
            else:
                self.LN = LTN(
                    num_features_in,
                    self.target_channels_list[0],
                    dropout=dropout,
                    bias=bias,
                    init_scheme=init_scheme,
                )
        else:
            if self.batch_cochain:

                self.LN_list = nn.ModuleList(
                    [
                        BatchLTN(
                            num_features_in,
                            target_channels_list[i],
                            dropout=dropout,
                            bias=bias,
                            init_scheme=init_scheme,
                        )
                        for i in range(0, len(target_channels_list))
                    ]
                )
            else:
                self.LN_list = nn.ModuleList(
                    [
                        LTN(
                            num_features_in,
                            target_channels_list[i],
                            dropout=dropout,
                            bias=bias,
                            init_scheme=init_scheme,
                        )
                        for i in range(0, len(target_channels_list))
                    ]
                )

    def forward(self, x: Tensor, G_list: list) -> list:
        r"""
        Args:
        ------
            x: torch tensor representing cell/simplicial features -
                x is of shape [num_in_cell, num_features_in]
            G_list - list of torch tensor representing a
                cochain matrices, each operator G_{j_k} in this list that
                represents a cochain map C^i->C^{j_k} .
                         Shape: [ num_target_feature_j, num_in_cell]
        output:
        -------
            list of pytorch tensors xi of length n
                 xi Shape : [num_target_cells,num_features_out]
        """
        if x is None:
            raise ValueError("input tensor cannot be None")
        if not self.shared_parameters:
            if len(G_list) != len(self.LN_list):
                raise ValueError("input lists must have the same lengths")

        for i in range(0, len(self.target_channels_list)):
            if G_list[i] is None:
                if self.target_channels_list[i] != self.num_features_in:
                    raise ValueError(
                        f"target channel dimension {i} must be equal to number of in",
                        f" feature since operator {i} is None",
                    )

        # TODO check operators are valid.
        # TODO assertion on the type of the inputs
        # TODO fix dropout
        if self.shared_parameters:
            output_tensors = [self.LN(x, G_list[i]) for i in range(0, len(G_list))]
        else:
            output_tensors = [
                self.LN_list[i](x, G_list[i]) for i in range(0, len(G_list))
            ]
        return output_tensors


class MultiSplitOper(_MultiSplitOper):
    r"""
    Description
    -----------
        Given a cochain x in :math:`C^i(X)`, the split operator sends x to n
        differnt cochains spaces :math:`C^{j_k}(X)`
        using n linear maps
        :math:`G_1:C^i(X) \to C^{j_k}(X)` for :math`0<=k<n`.
        Args
        ----
            num_features_in : number of features in the input channel
            target_channels_list : list of number of output features channels
            dropout: optional, default is 0.1
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_channels_list: list,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):
        super(MultiSplitOper, self).__init__(
            num_features_in,
            target_channels_list,
            shared_parameters,
            dropout,
            bias,
            init_scheme,
            False,
        )


class BatchMultiSplitOper(_MultiSplitOper):
    r"""
    Description
    -----------
        Given a cochain x in :math:`C^i(X)`, the split operator sends x to n
        differnt cochains spaces :math:`C^{j_k}(X)`
        using n linear maps
        :math:`G_1:C^i(X) \to C^{j_k}(X)` for :math`0<=k<n`.
        Args
        ----
            num_features_in : number of features in the input channel
            target_channels_list : list of number of output features channels
            dropout: optional, default is 0.1
            bias: optional, default is True
            init_scheme: optional, default is xavier, other options : debug."""

    def __init__(
        self,
        num_features_in: int,
        target_channels_list: list,
        shared_parameters=False,
        dropout=0.0,
        bias=True,
        init_scheme="xavier_uniform",
    ):

        super(BatchMultiSplitOper, self).__init__(
            num_features_in,
            target_channels_list,
            shared_parameters,
            dropout,
            bias,
            init_scheme,
            batch_cochain=True,
        )
