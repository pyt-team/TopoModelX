"""Multi-head Attention.

This module proposes two functionaly identical versions, one sparse and one dense implementations.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from .hoan import HigherOrderAttentionLayer, SparseHigherOrderAttentionLayer


class MultiHeadHigherOrderAttention(nn.Module):
    """Multi-head High-Order Attention.

    This is the dense implementation.

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
    num_heads : int, optional
        Number of attention heads, by default 5.
    alpha : float, optional
        Alpha value for the leaky_relu, by default 0.1.
    concatenate : bool, optional
        Whether to concatenate or average the attention heads, by default True.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        alpha=0.1,
        concatenate=True,
    ):
        super(MultiHeadHigherOrderAttention, self).__init__()
        self.concatenate = concatenate
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
        """Forward pass.

        This layer is inspired by:
        https://github.com/psh150204/GAT/blob/master/layer.py

        Parameters
        ----------
        input1 : torch.tensor, shape=[n_in_cell, source_in_features]
            Input features for the source cells.
        input2: torch.tensor, shape=[n_target_cell, target_in_features]
            Input features for the target cells.
        A_opt: torch.tensor, shape=[n_target_cell, n_in_cell]
            This can be the adjacency, laplacian matrix when the operator is symmetric.
            This can be a boundary matrix when the operator is asymmetric.

        Returns
        -------
        h_prime_st: torch.tensor, shape=[n_in_cell, source_out_features * num_heads]
            Output features for the source cells.
        h_prime_ts: torch.tensor, shape=[n_target_cell, target_out_features * num_heads]
            Output features for the target cells.
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

        if self.concatenate:
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
            return outputs_st / len(self.attentions), None


class SpMultiHeadHigherOrderAttention(nn.Module):
    """Multi-head High-Order Attention.

    This is the sparse implementation.

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
    num_heads : int, optional
        Number of attention heads, by default 5.
    alpha : float, optional
        Alpha value for the leaky_relu, by default 0.1.
    concatenate : bool, optional
        Whether to concatenate or average the attention heads, by default True.
    """

    def __init__(
        self,
        source_in_features,
        target_in_features,
        source_out_features,
        target_out_features,
        num_heads=5,
        alpha=0.1,
        concatenate=True,
    ):
        super(SpMultiHeadHigherOrderAttention, self).__init__()
        self.concatenate = concatenate
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
        """Forward pass.

        Parameters
        ----------
        input1 : torch.tensor, shape=[n_in_cell, source_in_features]
            Input features for the source cells.
        input2: torch.tensor, shape=[n_target_cell, target_in_features]
            Input features for the target cells.
        A_opt: torch.tensor, shape=[n_target_cell, n_in_cell]
            This can be the adjacency, laplacian matrix when the operator is symmetric.
            This can be a boundary matrix when the operator is asymmetric.
        operator_symmetry: bool
            Indicates if the A is symmetric or not.

        Returns
        -------
        h_prime_st: torch.tensor, shape=[n_in_cell, source_out_features * num_heads]
            Output features for the source cells.
        h_prime_ts: torch.tensor, shape=[n_target_cell, target_out_features * num_heads]
            Output features for the target cells.
        """

        if self.concatenate:
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


class MultiHeadHigherOrderAttentionClassifier(nn.Module):
    """Multi-head Attention Classifier.

    This the dense implementation.

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
    num_heads : int, optional
        Number of attention heads, by default 5.
    source_n_classes : int, optional
        Number of classes for the source cells, by default 5.
    target_n_classes : int, optional
        Number of classes for the target cells, by default 5.
    alpha : float, optional
        Alpha value for the leaky_relu, by default 0.1.
    """

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
        super(MultiHeadHigherOrderAttentionClassifier, self).__init__()
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
        """Forward pass.

        Inspired from: https://github.com/psh150204/GAT/blob/master/layer.py.

        Parameters
        ----------
        input1 : torch.tensor, shape=[n_in_cell, source_in_features]
            Input features for the source cells.
        input2: torch.tensor, shape=[n_target_cell, target_in_features]
            Input features for the target cells.
        A_opt: torch.tensor, shape=[n_target_cell, n_in_cell]
            This can be the adjacency, laplacian matrix when the operator is symmetric.
            This can be a boundary matrix when the operator is asymmetric.

        Returns
        -------
        h_prime_st: torch.tensor, shape=[n_in_cell, source_out_features * num_heads]
            Output features for the source cells.
        h_prime_ts: torch.tensor, shape=[n_target_cell, target_out_features * num_heads]
            Output features for the target cells.
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


class SpMultiHeadHigherOrderAttentionClassifier(nn.Module):
    """Multi-head Attention Classifier.

    This the sparse implementation.

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
    num_heads : int, optional
        Number of attention heads, by default 5.
    source_n_classes : int, optional
        Number of classes for the source cells, by default 5.
    target_n_classes : int, optional
        Number of classes for the target cells, by default 5.
    alpha : float, optional
        Alpha value for the leaky_relu, by default 0.1.
    """

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
        super(SpMultiHeadHigherOrderAttentionClassifier, self).__init__()

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
        """Forward pass.

        Inspired from: https://github.com/psh150204/GAT/blob/master/layer.py.

        Parameters
        ----------
        input1 : torch.tensor, shape=[n_in_cell, source_in_features]
            Input features for the source cells.
        input2: torch.tensor, shape=[n_target_cell, target_in_features]
            Input features for the target cells.
        A_opt: torch.tensor, shape=[n_target_cell, n_in_cell]
            This can be the adjacency, laplacian matrix when the operator is symmetric.
            This can be a boundary matrix when the operator is asymmetric.
        operator_symmetry: bool
            Indicates if the A is symmetric or not.

        Returns
        -------
        h_prime_st: torch.tensor, shape=[n_in_cell, source_out_features * num_heads]
            Output features for the source cells.
        h_prime_ts: torch.tensor, shape=[n_target_cell, target_out_features * num_heads]
            Output features for the target cells.
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
