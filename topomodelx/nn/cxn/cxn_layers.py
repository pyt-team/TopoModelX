"""Layers for CXN networks.

This implements four versions of message passing schemes defined on
regular cell complexes or simplicial complexes.

The implementation is based
on message passing schemes given in Cell Complex Neural Networks (CXNs) :
https://openreview.net/pdf?id=6Tq18ySFpGU
published in Topological Data Analysis and Beyond Workshop
at the 34th Conferenceon Neural Information Processing Systems
(NeurIPS 2020), Vancouver, Canada.
Released on Arxiv in Oct/2/2020.

These geometric message passing schemes are :
1) Adjacency message passing scheme networks AMPS.
2) Coadjacency message passing scheme networks CMPS.
3) Homology/Cohomology message passing scheme network HCMPS.
4) A general CXN implementation which considers all variations of the
    above message passing protocols is implemented in
    the class CXNGeneralLayer

The implementation assumes general regular
cell complex/simplicial complex of any dimension.
"""

__all__ = ["CXNGeneralLayer", "CXN_AMPS", "CXN_CMPS", "CXN_HCMPS"]
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.nn.cccnn import LTN, MergeOper, MultiMergeOper


class CXNGeneralLayer(nn.Module):
    r"""Class for a general CXN layer.

    A general CXN layer considers a general message passing
    that merge (co)adjacency and (co)boundary operators.

                   j_out
                 / | \   \
           Gi2j /  |  \   \  Gk2j
               /   |   \   \
              /Gj2j|    \   \
            i_in j1_in j2_in k_in

    Parameters
    ----------
    in_ch_i : int
    in_ch_j1 : int
    in_ch_j2 : int
    in_ch_k : int
    target_ch_j : int
    dropout : float, optional
        Dropout rate. The default is 0.05.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print verbose information. The default is False.
    """

    def __init__(
        self,
        in_ch_i,
        in_ch_j1,
        in_ch_j2,
        in_ch_k,
        target_ch_j,
        dropout=0.0,
        activation=F.relu,
        verbose=False,
    ):
        super(CXNGeneralLayer, self).__init__()

        self.in_ch_i = in_ch_i
        self.in_ch_j1 = in_ch_j1
        self.in_ch_j2 = in_ch_j2
        self.in_ch_k = in_ch_k
        self.target_ch_j = target_ch_j
        self.dropout = dropout
        self.multi_merge = MultiMergeOper(
            [in_ch_i, in_ch_j1, in_ch_j2, in_ch_k], target_ch_j
        )
        self.act = activation
        self.verbose = verbose

    def forward(self, xi, xj1, xj2, xk, Gi2j, Adj2j, coAdj2j, Gk2j):
        """Forward pass.

        Parameters
        ----------
        xi : torch.tensor, shape=[n_i_cells, n_i_features]
            Input feature vector on the i-cells of the input SC/CX.
        xj1 : torch.tensor, shape=[n_j_cells, n_j_features]
            Input feature vector on the j-cells of the input SC/CX.
        xj2 : torch.tensor, shape=[n_j_cells, n_j_features]
            Input feature vector on the j-cells of the input SC/CX.
        xk : torch.tensor, shape=[n_k_cells, num_k_features]
            Input feature vector on the k-cells of the input SC/CX.
        Gi2j : torch.tensor, shape=[n_j_cells , n_i_cells]
            Cochain operator C^i->C^j.
            Typically, a boundary matrix.
        Adj2j : torch.tensor, shape=[n_j_cells , n_i_cells]
            Cochain operator C^i->C^j.
        coAdj2j : torch.tensor, shape=[n_j_cells , n_j_cells]
            Cochain operator C^j->C^j.
            Typically, a higher order adj/Hodge Laplacian matrix.
        Gk2j : torch.tensor, shape=[n_j_cells , n_k_cells]
            Cochain operator C^k->C^j.
            Typically, a boundary matrix.

        Returns
        -------
        zj_out : torch.tensor, shape=[n_j_cells, target_ch_j]
            Output feature vector on the j-cells of the input SC/CX.
        """
        if set([xj1, xj2, xi, xk]) is {None}:
            raise ValueError("One of the input tensors must be non-empty.")

        if xj1 is not None:
            xj1 = F.dropout(xj1, self.dropout, self.training)
        if xj2 is not None:
            xj2 = F.dropout(xj2, self.dropout, self.training)
        if xi is not None:
            xi = F.dropout(xi, self.dropout, self.training)
        if xk is not None:
            xk = F.dropout(xk, self.dropout, self.training)

        z_j_final = self.multi_merge([xi, xj1, xj1, xk], [Gi2j, Adj2j, coAdj2j, Gk2j])
        if self.act is not None:
            z_j_final = self.act(z_j_final)

        return z_j_final


class CXN_AMPS(nn.Module):
    r"""Layer for adjacency message passing scheme (AMPS).

    AMPS has two main characteristics :
    (1) it uses the adjacency/Hodge Laplacian matrices to pass signal
    among cells/simplices of the same dimension.
    That is, a cochain map between C^i->C^i.
    (2) it uses boundary maps to pass signal down.
    In the figure below, signal moves from edges->nodes,
    and from faces->edges via the first and second
    boundary maps respectively.

    AMPS can be summarized via the figure below.
    Note that you need 2 maps to define an AMPS.

    Furthermore, you need two merge operations :
        (a) (v,e)->v and (b) (e,f)->e

           i_out
            | \
            |  \
        Gi2i|   \ Gj2i
            |    \
           i_in  j_in

    Parameters
    ----------
    in_ch_i : int
    in_ch_j : int
    target_ch_i : int
    dropout : float, optional
        Dropout rate. The default is 0.05.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print verbose information. The default is False.
    """

    def __init__(
        self,
        in_ch_i,
        in_ch_j,
        target_ch_i,
        dropout=0.1,
        activation=F.relu,
        verbose=False,
    ):
        super(CXN_AMPS, self).__init__()

        self.in_ch_i = in_ch_i
        self.in_ch_j = in_ch_j
        self.target_ch_i = target_ch_i
        self.dropout = dropout

        self.LTN1 = LTN(in_ch_i, target_ch_i)
        self.LTN2 = LTN(in_ch_j, target_ch_i)

        self.act = activation

        self.verbose = verbose

    def forward(self, xi, Gi2i, xj=None, Gj2i=None):
        """Forward pass.

        Parameters
        ----------
        xi : torch.tensor, shape=[n_i_cells, n_i_features]
            Input feature vector on the i-cells of the input SC/CX.
        xj : torch.tensor, shape=[n_j_cells, n_j_features]
            Input feature vector on the j-cells of the input SC/CX.
        Gi2i : torch.tensor, shape=[n_i_cells, n_i_cells]
            Cochain operator C^i->C^i.
        Gj2i : torch.tensor, shape=[n_i_cells, n_j_cells]
            Cochain operator C^i->C^j.
            Typically, a boundary matrix.

        Returns
        -------
        zi_out : torch.tensor, shape=[n_i_cells, target_ch_i]
            Feature vector on the i-cells of the input SC/CX.
        """
        xi = F.dropout(xi, self.dropout, self.training)

        if xj is not None:
            xj = F.dropout(xj, self.dropout, self.training)

        zi_out = self.LTN1(xi, Gi2i)

        if Gj2i is not None and xj is not None:
            zj_out = self.LTN2(xj, Gj2i)

        z_i_final = zi_out + zj_out

        if self.act is not None:
            z_i_final = self.act(z_i_final)

        return z_i_final


class CXN_CMPS(nn.Module):
    """Layer for coadjacency message passing scheme (CMPS).

    CMPS has two main characteristics :
    (1) it uses the coadjacency/Hodge Laplacian matrices
        to move signal among simplices of the same dimension.
        That is, a cochain map between C^i->C^i.
    (2) it uses coboundary maps to move signal up (in the diagonal part).
        In the figure below, signal moves from nodes->edges, and from
        edges->faces via the first and second
        coboundary maps respectively.

              j_out
               /|
              / |
        Gi2j /  | Gj2j
            /   |
          i_in  j_in

    Parameters
    ----------
    in_ch_i : int
    in_ch_j : int
    target_ch_j : int
    dropout : float, optional
        Dropout rate. The default is 0.05.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print verbose information. The default is False.
    """

    def __init__(
        self, in_ch_i, in_ch_j, target_ch_j, dropout=0.1, activation=None, verbose=False
    ):
        super(CXN_CMPS, self).__init__()

        self.CMPS = CXN_AMPS(
            in_ch_i=in_ch_j,
            in_ch_j=in_ch_i,
            target_ch_i=target_ch_j,
            dropout=dropout,
            activation=activation,
            verbose=verbose,
        )

    def forward(self, xj, xi, Gj2j, Gi2j):
        """Forward pass.

        Parameters
        ----------
        xj : torch.tensor, shape=[n_j_cells, n_j_features]
            Input feature vector on the j-cells of the input SC/CX.
        xi : torch.tensor, shape=[n_i_cells, n_i_features]
            Input feature vector on the i-cells of the input SC/CX.
        Gj2j : torch.tensor, shape=[n_j_cells, n_j_cells]
            Cochain operator C^j -> C^j.
        Gi2j : torch.tensor, shape=[n_j_cells , n_i_cells]
            Cochain operator C^i -> C^j.
            Typically, a coboundary matrix.

        Returns
        -------
        zj_out : torch.tensor, shape=[batch_size, n_nodes, target_ch]
            Feature vector on the nodes of the input SC/CX.
        """
        zj_out = self.CMPS(xj, xi, Gj2j, Gi2j)

        return zj_out


class CXN_HCMPS(nn.Module):
    r"""Layer for the homology/cohomology message passing scheme (HCMPS).

    HCMPS has a main characteristic :
    (*) it uses only boundary and coboundary maps
        to move signal up and down (in the diagonal part).

    In the case of 2d simplicial complex, HCMPS can be
    summarized via the figure below. Note that you need
    2 maps to define an HCMPS.

    Two of these maps are the boundary maps and the other
    two are simply the tranpose of these maps.
    Furthermore, you need one merge operation and one split
    operation : (a) (v,f)->e and (b) (e)->(v,f)

                 k_out
                 /   \
         Gj2k   /     \  Gi2k
               /       \
              /         \
           j_in         i_in

    Parameters
    ----------
    in_ch_i : int
    in_ch_j : int
    target_ch_k : int
    dropout : float, optional
        Dropout rate. The default is 0.05.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print verbose information. The default is False.
    """

    def __init__(
        self,
        in_ch_i,
        in_ch_j,
        target_ch_k,
        dropout=0.5,
        activation=F.relu,
        verbose=False,
    ):
        super(CXN_HCMPS, self).__init__()

        self.dropout = dropout
        self.merge_i_j = MergeOper(in_ch_i, in_ch_j, target_ch_k)  # (i,j)->k
        self.act = activation
        self.verpose = verbose

    def forward(self, xi, xj, Gi2k, Gj2k):
        """Forward pass.

        Parameters
        ----------
        xi : torch.tensor, shape=[n_i_cells, n_i_features]
            Input feature vector on the nodes of the input SC/CX
        xj : torch.tensor, shape=[n_j_cells, n_j_features]
            Input feature vector on the edges of the input SC/CX
        Gi2k : torch.tensor, shape=[n_k_cells, n_i_cells]
            Cochain operator C^i-> C^k.
        Gj2k : torch.tensor, shape=[n_k_cells ,n_j_cells]
            Cochain operator C^j -> C^k.

        Returns
        -------
        zk_out: torch.tensor, shape=[n_nodes, target_ch_v]
            Feature vector on the k-cells of the input SC/CX.
        """
        if xi is not None:
            xi = F.dropout(xi, self.dropout, self.training)
        if xj is not None:
            xj = F.dropout(xj, self.dropout, self.training)
        zk_out = self.merge_i_j(xi, xj, Gi2k, Gj2k)  # i->k , j->k

        if self.act is not None:
            zk_out = self.act(zk_out)
        return zk_out
