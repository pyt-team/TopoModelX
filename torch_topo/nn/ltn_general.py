# -*- coding: utf-8 -*-
"""

@author: Mustafa Hajij
"""

from warnings import warn

import torch
import torch.nn as nn
import torch.nn.functional as F
from stnets.layers.linear import Linear
from stnets.layers.message_passing import HigherOrderMessagePassing
from stnets.util import batch_mm
from torch import Tensor
from torch.nn.parameter import Parameter


class _LTNMessagePassing(HigherOrderMessagePassing):
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
        activation=torch.nn.ReLU(),
        init_scheme="xavier_uniform",
        batch_cochain=True,
    ):
        super(_LTNMessagePassing, self).__init__()

        self.in_ft = in_ft
        self.out_ft = out_ft
        self.init_scheme = init_scheme
        self.batch_cochain = batch_cochain
        self.dropout = dropout
        self.activation = activation

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
                    [ num_in_cell, num_features_in]
            if batch_cochain is False:
                x : cellular/simplicial features - Tensor with cell
                    features of shape [num_in_cell, num_features_in]
            A_operator : a cochain matrix that represents a
            cochain map C^i->C^j . Entry A_operator[i,j]=1 means there
            is a message from cell/simplex i to cell/simplex j .
        output:
        -------
            pytorch tensor x:
                 Shape : [num_cells_out,num_features_out ]

        """
        if self.in_ft != self.out_ft:
            raise ValueError(
                "The input operator is None or 'Id' acts"
                + " as an identity operator,"
                + "the in_ft must be the same as out_ft "
                + "in the model constructor."
            )
        if self.in_ft != x.shape[-1]:
            raise ValueError(
                "The input operator is None acts as an identity "
                + "operator, the in_ft must be the same as number"
                + " of features in the input cochain"
            )

        if not isinstance(A_operator, torch.Tensor):
            raise TypeError(
                f"Input operator must be torch tensor, instead got an input of type {type(A_operator)}."
            )
        if not isinstance(x, torch.Tensor):
            raise TypeError(
                f"Input cochain must be torch tensor, instead got an input of type {type(x)}."
            )

        if (
            len(x.shape) == 2
        ):  # Assuming single input, batchsize=1 and no batch channel is included
            if x.shape[0] != A_operator.shape[-1]:
                raise ValueError(
                    "number of source cells/simplicies must match number of elements in input tensor."
                )

        x = x @ self.weight
        if self.bias is not None:
            x = self.bias + x
        x = F.dropout(x, self.dropout, self.training)
        x = self.activation(x)

        return self.propagate(x, A_operator)
