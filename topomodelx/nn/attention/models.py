# -*- coding: utf-8 -*-
"""
Created on Wed Jan 12 16:16:44 2022

@author: Mustafa Hajij
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hoan import HigherOrderAttentionLayer, SparseHigherOrderAttentionLayer

"""
    Simplicial/Cellular/hypergraph Mutli-head Attention , sparse and dense implementations
"""


class MultiHeadHigherOrderAttention(nn.Module):
    """
    Simplicial/Cellular Mutli-head Attention
    """

    # multi head attention
    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        alpha=0.1,
        concatinate=True,
    ):
        super(MultiHeadHigherOrderAttention, self).__init__()
        self.concatinate = concatinate
        self.attentions = nn.ModuleList(
            [
                HigherOrderAttentionLayer(
                    source_in_features=source_in_features,
                    target_in_features=target_in_features,
                    source_out_features=source_out_features,
                    target_out_features=target_out_features,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, input1, input2, A_opt):
        """
        Parameters:

            input1. A torch tensor of shape [num_in_cell, num_source_in_features]
            input2. A torch tensor of shape [num_target_cell, num_target_in_features]
            A_opt.  A torch tensor of shape [num_target_cell, num_in_cell]. Typically this can be a boundary matrix when the operator is asymmetrical, the adjacency, laplacian matrix when the operator is symmetric.

        Return :

            h_prime_st. A torch tensor of shape [num_in_cell, num_source_out_features * num_heads]
            h_prime_ts. A torch tensor of shape [num_target_cell, num_target_out_features * num_heads]
        """

        """
        layer built based on  https://github.com/psh150204/GAT/blob/master/layer.py
        """

        if input2 is None:
            if A_opt.shape[0] != A_opt.shape[1]:
                raise ValueError(
                    "The operator A must be symmetric when the target features are None."
                )
        else:
            if A_opt.shape[0] == A_opt.shape[1]:
                raise ValueError(
                    "the target features cannot must be None when the operator A is symmetric."
                )

        if self.concatinate:
            # concatenate
            outputs_st = []
            outputs_ts = []
            for attention in self.attentions:
                hst, hts = attention(input1, input2, A_opt)
                outputs_st.append(hst)
                if input2 is not None:
                    outputs_ts.append(hts)
            if input2 is not None:
                return torch.cat(outputs_st, dim=-1), torch.cat(outputs_ts, dim=-1)
            else:
                return torch.cat(outputs_st, dim=-1), None

        else:
            # average
            outputs_st = None
            outputs_ts = None
            for attention in self.attentions:
                if outputs_st is None:
                    outputs_st, outputs_ts = attention(input1, input2, A_opt)
                else:
                    hst, hts = attention(input1, input2, A_opt)
                    outputs_st = outputs_st + hst
                    if (
                        hts is not None
                    ):  # operator A_opt is symmetric -> second feature vector is None
                        outputs_ts = outputs_ts + hts

            if input2 is not None:
                return (
                    outputs_st / len(self.attentions),
                    outputs_ts / len(self.attentions),
                )
            else:
                return outputs_st / len(self.attentions), None


class SpMultiHeadHigherOrderAttention(nn.Module):
    """
    Simplicial/Cellular Mutli-head Attention
    """

    # multi head attention
    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        alpha=0.1,
        concatinate=True,
    ):
        super(SpMultiHeadHigherOrderAttention, self).__init__()
        self.concatinate = concatinate
        self.attentions = nn.ModuleList(
            [
                SparseHigherOrderAttentionLayer(
                    source_in_features=source_in_features,
                    target_in_features=target_in_features,
                    source_out_features=source_out_features,
                    target_out_features=target_out_features,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(num_heads)
            ]
        )

    def forward(self, input1, input2, A_opt, operator_symmetry=False):
        """
        Parameters:

            -input1. A torch tensor of shape [num_in_cell, num_source_in_features]
            -input2. A torch tensor of shape [num_target_cell, num_target_in_features]

            -operator_list. A torch tensor of size represents the cochain map C^s -> C^t.
             Operator_list expect a sparse rep of this operator and in this case we use the sparse rep to be a torch tensor  of
             the shape [2, K] where K is the number of non-zero elements dense matrix corresoponds to the sparse array.
             First row in this matrix is the rows_indices of the sparse rep and the second row is the column_indices
             In other words, if there is a message between cell i and a cell j then there the sparse representation that stores the indices i and j.
             For exampl the dense operator A_opt= tensor([[1., 1., 0.],
                                                         [0., 1., 1.]]) can be written as
                                       operator_list= tensor([[0, 0, 1, 1],
                                                              [0, 1, 1, 2]]) which stores the positions of the nonzero elements in the (dense representation ) of the operator_list
             Typically operator_list  can be a boundary matrix when the operator is asymmetrical,
             the adjacency, laplacian matrix when the operator is symmetric.

            -operator_symmetry. bool, indicating if the is symmetric or not.

        Return :

            -h_prime_st. A torch tensor of shape [num_in_cell, num_source_out_features * num_heads]
            -h_prime_ts. A torch tensor of shape [num_target_cell, num_target_out_features * num_heads]
        """

        if self.concatinate:
            # concatenate
            outputs_st = []
            outputs_ts = []
            for attention in self.attentions:
                hst, hts = attention(input1, input2, A_opt, operator_symmetry)
                outputs_st.append(hst)
                if input2 is not None:
                    outputs_ts.append(hts)
            if input2 is not None:
                return torch.cat(outputs_st, dim=-1), torch.cat(outputs_ts, dim=-1)
            else:
                return torch.cat(outputs_st, dim=-1), None

        else:
            # average
            outputs_st = None
            outputs_ts = None
            for attention in self.attentions:
                if outputs_st is None:
                    outputs_st, outputs_ts = attention(
                        input1, input2, A_opt, operator_symmetry
                    )
                else:
                    hst, hts = attention(input1, input2, A_opt, operator_symmetry)
                    outputs_st = outputs_st + hst
                    if (
                        hts is not None
                    ):  # operator A_opt is symmetric -> second feature vector is None
                        outputs_ts = outputs_ts + hts

            if input2 is not None:
                return (
                    outputs_st / len(self.attentions),
                    outputs_ts / len(self.attentions),
                )
            else:
                return outputs_st / len(self.attentions), None


class MultiHeadHigherOrderAttentionClassifer(nn.Module):
    """
    Simplicial/Cellular Mutli-head Attention
    """

    # multi head attention
    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        source_n_classes=5,
        target_n_classes=5,
        alpha=0.1,
    ):
        super(MultiHeadHigherOrderAttentionClassifer, self).__init__()
        self.attentions = nn.ModuleList(
            [
                HigherOrderAttentionLayer(
                    source_in_features=source_in_features,
                    target_in_features=target_in_features,
                    source_out_features=source_out_features,
                    target_out_features=target_out_features,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_att = HigherOrderAttentionLayer(
            source_in_features=source_out_features * num_heads,
            target_in_features=target_out_features * num_heads,
            source_out_features=source_n_classes,
            target_out_features=target_n_classes,
            alpha=alpha,
            concat=False,
        )

    def forward(self, input1, input2, A_opt):
        """
        Parameters:

            input1. A torch tensor of shape [num_in_cell, num_source_in_features]
            input2. A torch tensor of shape [num_target_cell, num_target_in_features]
            A_opt.  A torch tensor of shape [num_target_cell, num_in_cell]. Typically this can be a boundary matrix when the operator is asymmetrical, the adjacency, laplacian matrix when the operator is symmetric.

        Return :

            h_prime_st. A torch tensor of shape [num_in_cell, num_source_out_features * num_heads]
            h_prime_ts. A torch tensor of shape [num_target_cell, num_target_out_features * num_heads]
        """

        """
        layer built based on  https://github.com/psh150204/GAT/blob/master/layer.py
        """

        if input2 is None:
            if A_opt.shape[0] != A_opt.shape[1]:
                raise ValueError(
                    "The operator A must be symmetric when the target features are None."
                )
        else:
            if A_opt.shape[0] == A_opt.shape[1]:
                raise ValueError(
                    "the target features cannot must be None when the operator A is symmetric."
                )

        outputs_st = []
        outputs_ts = []
        for attention in self.attentions:
            hst, hts = attention(input1, input2, A_opt)
            outputs_st.append(hst)
            if input2 is not None:
                outputs_ts.append(hts)
        if input2 is not None:
            outputs_st = torch.cat(outputs_st, dim=1)
            outputs_ts = torch.cat(outputs_ts, dim=1)

            outputs_st, outputs_ts = self.out_att(outputs_ts, outputs_st, A_opt)

            outputs_st = F.elu(outputs_st)
            outputs_ts = F.elu(outputs_ts)
            return F.log_softmax(outputs_st, dim=1), F.log_softmax(outputs_ts, dim=1)

        else:
            outputs_st = torch.cat(outputs_st, dim=1)

            outputs_st, _ = self.out_att(outputs_st, None, A_opt)
            outputs_st = F.elu(outputs_st)

            return F.log_softmax(outputs_st, dim=1), None


class SpMultiHeadHigherOrderAttentionClassifer(nn.Module):
    """
    Sparse Simplicial/Cellular Mutli-head Attention
    """

    # multi head attention
    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        source_n_classes=5,
        target_n_classes=5,
        alpha=0.1,
    ):
        super(SpMultiHeadHigherOrderAttentionClassifer, self).__init__()

        self.attentions = nn.ModuleList(
            [
                SparseHigherOrderAttentionLayer(
                    source_in_features=source_in_features,
                    target_in_features=target_in_features,
                    source_out_features=source_out_features,
                    target_out_features=target_out_features,
                    alpha=alpha,
                    concat=True,
                )
                for _ in range(num_heads)
            ]
        )
        self.out_att = SparseHigherOrderAttentionLayer(
            source_in_features=source_out_features * num_heads,
            target_in_features=target_out_features * num_heads,
            source_out_features=source_n_classes,
            target_out_features=target_n_classes,
            alpha=alpha,
            concat=False,
        )

    def forward(self, input1, input2, A_opt, operator_symmetry=False):
        """
        Parameters:

            -input1. A torch tensor of shape [num_in_cell, num_source_in_features]
            -input2. A torch tensor of shape [num_target_cell, num_target_in_features]

            -operator_list. A torch tensor of size represents the cochain map C^s -> C^t.
             Operator_list expect a sparse rep of this operator and in this case we use the sparse rep to be a torch tensor  of
             the shape [2, K] where K is the number of non-zero elements dense matrix corresoponds to the sparse array.
             First row in this matrix is the rows_indices of the sparse rep and the second row is the column_indices
             In other words, if there is a message between cell i and a cell j then there the sparse representation that stores the indices i and j.
             For exampl the dense operator A_opt= tensor([[1., 1., 0.],
                                                         [0., 1., 1.]]) can be written as
                                       operator_list= tensor([[0, 0, 1, 1],
                                                              [0, 1, 1, 2]]) which stores the positions of the nonzero elements in the (dense representation ) of the operator_list
             Typically operator_list  can be a boundary matrix when the operator is asymmetrical,
             the adjacency, laplacian matrix when the operator is symmetric.

            -operator_symmetry. bool, indicating if the is symmetric or not.

        Return :

            -h_prime_st. A torch tensor of shape [num_in_cell, num_source_out_features * num_heads]
            -h_prime_ts. A torch tensor of shape [num_target_cell, num_target_out_features * num_heads]
        """

        outputs_st = []
        outputs_ts = []
        for attention in self.attentions:
            hst, hts = attention(input1, input2, A_opt, operator_symmetry)
            outputs_st.append(hst)
            if input2 is not None:
                outputs_ts.append(hts)

        if input2 is not None:
            outputs_st = torch.cat(outputs_st, dim=1)
            outputs_ts = torch.cat(outputs_ts, dim=1)

            outputs_st, outputs_ts = self.out_att(
                outputs_ts, outputs_st, A_opt, operator_symmetry
            )

            outputs_st = F.elu(outputs_st)
            outputs_ts = F.elu(outputs_ts)
            return F.log_softmax(outputs_st, dim=1), F.log_softmax(outputs_ts, dim=1)

        else:
            outputs_st = torch.cat(outputs_st, dim=1)

            outputs_st, _ = self.out_att(outputs_st, None, A_opt, operator_symmetry)
            outputs_st = F.elu(outputs_st)

            return F.log_softmax(outputs_st, dim=1), None
