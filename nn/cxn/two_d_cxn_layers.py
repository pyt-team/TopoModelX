"""Cellular complex networks on a 2D regular complex.

This implements three versions of message passing schemes defined on
regular cell complexes or simplicial complexes. The implementation is based
on message passing schemes given in Cell Complex Neural Networks (CXNs) :
https://openreview.net/pdf?id=6Tq18ySFpGU
publishedin Topological Data Analysis and Beyond Workshop
at the 34th Conferenceon Neural Information Processing Systems
(NeurIPS 2020), Vancouver, Canada.
Released on Arxiv in Oct/2/2020.

These geometric message passing schemes are :
1) Adjacency message passing scheme networks AMPS.
2) Coadjacency message passing scheme networks CMPS.
3) Homology/Cohomology message passing scheme network HCMPS.
4) Also implemented the message passing function that combine
    the above schemes all together.

The implementation here is given on a 2d regular cell complex
(see our paper above for definition) or 2d simplicial complex.
In particular, the implementation here is applicable to
triangular and polygonal surfaces/meshes.
"""

__all__ = ["two_d_CXN", "two_d_CXN_AMPS", "two_d_CXN_CMPS", "two_d_CXN_HCMPS"]

import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.nn.cccnn import LTN, MergeOper, MultiMergeOper, SplitOper


class two_d_CXN(nn.Module):
    r"""General message passing on 2d regular cell complex.

    v_ot   e_ot   f_ot
    | \  / | |   \  / |
    |  \/  | |    \/  |
    | /  \ | |   /  \ |
    |/    \| |  /    \|
    v_in  e_in  f_in

    Parameters
    ----------
    in_ch_v : int
        Number of input channels for vertices.
    in_ch_e : int
        Number of input channels for edges.
    in_ch_f : int
        Number of input channels for faces.
    target_ch_v : int
        Number of output channels for vertices.
    target_ch_e : int
        Number of output channels for edges.
    target_ch_f : int
        Number of output channels for faces.
    dropout : float, optional
        Dropout rate. The default is 0.5.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print out the shape of the output of each layer. The default is False.
    merge_type : str, optional
        Type of merge operation. The default is "average".
    shared_parameters : bool, optional
        Whether to share parameters between merge operations. The default is False.
    """

    def __init__(
        self,
        in_ch_v,
        in_ch_e,
        in_ch_f,
        target_ch_v,
        target_ch_e,
        target_ch_f,
        dropout=0.5,
        activation=F.relu,
        verbose=False,
        merge_type="average",
        shared_parameters=False,
    ):
        super(two_d_CXN, self).__init__()
        self.dropout = dropout

        self.merge_v_e = MergeOper(
            in_ch_v,
            in_ch_e,
            target_ch_v,
            merge_type=merge_type,
            shared_parameters=shared_parameters,
        )  # merge vertices and edges features

        self.merge_v_e_f = MultiMergeOper(
            [in_ch_v, in_ch_e, in_ch_e, in_ch_f],
            target_ch_e,
            merge_type=merge_type,
            shared_parameters=shared_parameters,
        )  # merge nodes edges and faces features

        self.merge_e_f = MergeOper(
            in_ch_e,
            in_ch_f,
            target_ch_f,
            merge_type=merge_type,
            shared_parameters=shared_parameters,
        )  # merge edges and faces features

        self.act = activation

        self.verbose = verbose

    def forward(self, xv, xe, xf, Gv2v, Ge2v, Ge2e, He2e, Gf2e, Gf2f):
        """Forward pass.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Input feature vector on the nodes of the input SC/CX.
        xe : torch tensor, shape=[n_edges, n_edges_features]
            Input feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Input feature vector on the faces of the input SC/CX.
        Gv2v : torch.tensor, shape=[n_nodes, n_nodes]
            Cochain operator C^0->C^0.
        Ge2v : torch.tensor, shape=[n_nodes, n_edges]
            Cochain operator C^1->C^0.
        Ge2e : torch.tensor, shape=[n_edges, n_edges]
            Cochain operator C^1->C^1.
        He2e : torch.tensor, shape=[n_edges, n_edges]
            Cochain operator C^1->C^1.
        Gf2e : torch.tensor, shape=[n_edges, n_faces]
            Cochain operator C^2->C^1.
        Gf2f : torch.tensor, shape=[n_faces, n_faces]
            Cochain operator C^2->C^2.

        Returns
        -------
        zv_out : torch.tensor, shape=[n_nodes, target_ch_v]
            Output feature vector on the nodes of the input SC/CX.
        ze_out : torch.tensor, shape=[n_edges, target_ch_e]
            Output feature vector on the edges of the input SC/CX.
        zf_out : torch.tensor, shape=[n_edges, target_ch_f]
            Output feature vector on the faces of the input SC/CX.
        """
        if xv is not None:
            xv = F.dropout(xv, self.dropout, self.training)
        if xe is not None:
            xe = F.dropout(xe, self.dropout, self.training)
        if xf is not None:
            xf = F.dropout(xf, self.dropout, self.training)

        if Ge2v is None:
            v2e = None
        else:
            v2e = torch.t(Ge2v)

        if Gf2e is None:
            e2f = None
        else:
            e2f = torch.t(Gf2e)

        zv_out = self.merge_v_e(xv, xe, Gv2v, Ge2v)  # target is vertices
        ze_out = self.merge_v_e_f(
            [xv, xe, xe, xf], [v2e, Ge2e, He2e, Gf2e]
        )  # target is edges
        zf_out = self.merge_e_f(xe, xf, e2f, Gf2f)  # target is faces

        if self.act is not None:
            zv_out = self.act(zv_out)
            ze_out = self.act(ze_out)
            zf_out = self.act(zf_out)

        return zv_out, ze_out, zf_out


class two_d_CXN_AMPS(nn.Module):
    r"""Adjacency message passing scheme (AMPS) on a 2d regular complex.

    AMPS has two main characteristics:
    (1) it uses the adjacency/Hodge Laplacian
        matrices to move signal among simplices
        of the same dimension. That is, a cochain
        map between C^i->C^i
    (2) it uses boundary maps to move signal down.
        In the figure below, signal moves from
        edges->nodes, and from faces->edges via
        the first and second boundary maps respectively.

    In the case of 2d simplicial complex,
    AMPS can be summarized via the figure below.

    Note that you need 5 maps to define an AMPS.

    Furthermore, you need two merge operations:
    (a) (v, e)->v
    (b) (e,f)->e

    v_ot   e_ot   f_ot
    | \   | \   |
    |  \  |  \  |
    |   \ |   \ |
    |    \|    \|
    v_in  e_in  f_in

    Parameters
    ----------
    in_ch_v : int
        Number of input channels for vertices.
    in_ch_e : int
        Number of input channels for edges.
    in_ch_f : int
        Number of input channels for faces.
    target_ch_v : int
        Number of output channels for vertices.
    target_ch_e : int
        Number of output channels for edges.
    target_ch_f : int
        Number of output channels for faces.
    dropout : float, optional
        Dropout rate. The default is 0.5.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print out the shape of the output of each layer. The default is False.
    """

    def __init__(
        self,
        in_ch_v,
        in_ch_e,
        in_ch_f,
        target_ch_v,
        target_ch_e,
        target_ch_f,
        dropout=0.5,
        activation=F.relu,
        verbose=False,
    ):
        super(two_d_CXN_AMPS, self).__init__()
        self.dropout = dropout

        self.merge_v_e = MergeOper(
            in_ch_v, in_ch_e, target_ch_v
        )  # merge vertices and edges features

        self.merge_e_f = MergeOper(
            in_ch_e, in_ch_f, target_ch_e
        )  # merge edges and faces features

        self.LTNf2f = LTN(in_ch_f, target_ch_f)  # send face features to themselves

        self.act = activation

        self.verbose = verbose

    def forward(self, xv, xe, xf, Gv2v, Ge2v, Ge2e, Gf2e, Gf2f):
        """Forward pass of the AMPS.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Input feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[n_edges, n_edges_features]
            Input feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[ n_faces, n_faces_features]
            Input feature vector on the faces of the input SC/CX.
        Gv2v : torch.tensor, shape=[n_nodes, n_nodes]
            Cochain operator C^0->C^0.
        Ge2v : torch.tensor, shape=[n_nodes, n_edges]
            Cochain operator C^1->C^0.
        Ge2e : torch.tensor, shape=[n_edges, n_edges]
            Cochain operator C^1->C^1.
        Gf2e : torch.tensor, shape=[n_edges, n_faces]
            Cochain operator C^2->C^1.
        Gf2f : torch.tensor, shape=[n_faces, n_faces]
            Cochain operator C^2->C^2.

        Returns
        -------
        zv_out : torch.tensor, shape=[n_nodes, target_ch_v]
            Output feature vector on the nodes of the input SC/CX.
        ze_out : torch.tensor, shape=[n_edges, target_ch_e]
            Output feature vector on the edges of the input SC/CX.
        zf_out : torch.tensor, shape=[n_faces, target_ch_f]
            Output feature vector on the faces of the input SC/CX.
        """

        if xv is not None:
            xv = F.dropout(xv, self.dropout, self.training)
        if xe is not None:
            xe = F.dropout(xe, self.dropout, self.training)
        if xf is not None:
            xf = F.dropout(xf, self.dropout, self.training)
        zv_out = self.merge_v_e(xv, xe, Gv2v, Ge2v)  # target is vertices
        ze_out = self.merge_e_f(xe, xf, Ge2e, Gf2e)  # target is edges
        zf_out = self.LTNf2f(xf, Gf2f)  # target is faces

        if self.act is not None:
            zv_out = self.act(zv_out)
            ze_out = self.act(ze_out)
            zf_out = self.act(zf_out)

        return zv_out, ze_out, zf_out


class two_d_CXN_CMPS(nn.Module):
    r"""Coadjaceny message passing scheme (CMPS) for 2d regular complexes.

    CMPS has two main characteristics:
    (1) it uses the coadjacency/Hodge Laplacian matrices to move
        signal among simplices of the same dimension.
        That is, a cochain map between C^i->C^i.
    (2) it uses coboundary maps to move signal up
        (in the diagonal part).

    In the figure below,
    signal moves from nodes->edges, and from edges->faces
    via the first and second coboundary maps respectively.

    In the case of 2d simplicial complex, CMPS can be
    summarized via the figure below.

    Note that you need 5 maps to define an CMPS.

    Furthermore, you need two merge operations:
    (a) (v,e)->e
    (b) (e,f)->f.

    v_ot   e_ot   f_ot
     |    /|    / |
     |   / |   /  |
     |  /  |  /   |
     | /   | /    |
     v_in  e_in  f_in

    Parameters
    ----------
    in_ch_v : int
        Number of input channels for vertices.
    in_ch_e : int
        Number of input channels for edges.
    in_ch_f : int
        Number of input channels for faces.
    target_ch_v : int
        Number of output channels for vertices.
    target_ch_e : int
        Number of output channels for edges.
    target_ch_f : int
        Number of output channels for faces.
    dropout : float, optional
        Dropout rate. The default is 0.5.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print out the shape of the output of each layer. The default is False.
    """

    def __init__(
        self,
        in_ch_v,
        in_ch_e,
        in_ch_f,
        target_ch_v,
        target_ch_e,
        target_ch_f,
        dropout=0.1,
        activation=None,
        verbose=False,
    ):
        super(two_d_CXN_CMPS, self).__init__()
        self.dropout = dropout

        self.LTNv2v = LTN(in_ch_v, target_ch_v)

        self.merge_v_e = MergeOper(in_ch_v, in_ch_e, target_ch_e)

        self.merge_e_f = MergeOper(in_ch_e, in_ch_f, target_ch_f)

        self.act = activation

        self.verbose = verbose

    def forward(self, xv, xe, xf, Gv2v, Gv2e, Ge2e, Ge2f, Gf2f):
        """Forward pass of the CMPS.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Input feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[n_edges, n_edges_features]
            Input feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Input feature vector on the faces of the input SC/CX.
        Gv2v : torch.tensor, shape=[n_nodes, n_nodes]
            Cochain operator C^0 -> C^0.
        Gv2e : torch.tensor, shape=[n_edges, n_nodes]
            Cochain operator C^0 -> C^1.
        Ge2e : torch.tensor, shape=[n_edges, n_edges]
            Cochain operator C^1 -> C^1.
        Ge2f : torch.tensor, shape=[n_faces, n_edges]
            Cochain operator C^1 -> C^2.
        Gf2f : torch.tensor, shape=[n_faces, n_faces]
            Cochain operator C^2 -> C^2.

        Returns
        -------
        zv_out : torch.tensor, shape=[n_nodes, target_ch ]
            Output feature vector on the nodes of the input SC/CX.
        ze_out : torch.tensor, shape=[n_edges, target_ch ]
            Output feature vector on the edges of the input SC/CX.
        zf_out : torch.tensor, shape=[n_edges, target_ch]
            Output feature vector on the faces of the input SC/CX.
        """
        if xv is not None:
            xv = F.dropout(xv, self.dropout, self.training)
        if xe is not None:
            xe = F.dropout(xe, self.dropout, self.training)
        if xf is not None:
            xf = F.dropout(xf, self.dropout, self.training)

        zv_out = self.LTNv2v(xv, Gv2v)  # target is vertices
        ze_out = self.merge_v_e(xv, xe, Gv2e, Ge2e)  # target is edges
        zf_out = self.merge_e_f(xe, xf, Ge2f, Gf2f)  # target is faces

        if self.activation is not None:
            zv_out = self.act(zv_out)
            ze_out = self.act(ze_out)
            zf_out = self.act(zf_out)

        return zv_out, ze_out, zf_out


class two_d_CXN_HCMPS(nn.Module):
    r"""Homology/Co-homology message passing scheme (HCMPS) layer.

    HCMPS has one main characteristic:
    (*) it uses only boundary and coboundary
        maps to move signal up and down (in the diagonal part).

    In the case of 2d simplicial complex,
    HCMPS can be summarized via the figure below.

    Note that you need 2 maps to define an HCMPS.
    Two of these maps are the boundary maps and the
    other two are simply the tranpose of these maps.

    Furthermore, you need one merge operation and
    one split operation :
    (a) (v,f)->e
    (b) (e)->(v,f)

    v_ot  e_ot   f_ot
       \  /   \ /
        \/    /\
        /\   /  \
       /  \ /    \
    v_in  e_in  f_in

    Parameters
    ----------
    in_ch_v : int
        Number of input channels for vertices.
    in_ch_e : int
        Number of input channels for edges.
    in_ch_f : int
        Number of input channels for faces.
    target_ch_v : int
        Number of output channels for vertices.
    target_ch_e : int
        Number of output channels for edges.
    target_ch_f : int
        Number of output channels for faces.
    dropout : float, optional
        Dropout rate. The default is 0.5.
    activation : torch.nn.functional, optional
        Activation function. The default is F.relu.
    verbose : bool, optional
        Print out the shape of the output of each layer. The default is False.
    """

    def __init__(
        self,
        in_ch_v,
        in_ch_e,
        in_ch_f,
        target_ch_v,
        target_ch_e,
        target_ch_f,
        dropout=0.5,
        activation=None,
        verbose=False,
    ):
        super(two_d_CXN_HCMPS, self).__init__()
        self.dropout = dropout
        self.merge_v_f = MergeOper(in_ch_v, in_ch_f, target_ch_e)  # (v,f)->e
        self.split_v_f = SplitOper(in_ch_e, target_ch_v, target_ch_f)  # e->(v,f)
        self.act = activation
        self.verpose = verbose

    def forward(self, xv, xe, xf, Ge2v, Gf2e):
        """Forward pass of the HCMPS.

        Parameters
        ----------
        xv : torch.tensor, shape=[batch_size, n_nodes, n_node_features]
            Input feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[n_edges, n_edges_features]
            Input feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Input feature vector on the faces of the input SC/CX.
        Ge2v : torch.tensor, shape=[n_nodes, n_edges]
            Cochain operator C^1-> C^0.
        Gf2e : torch.tensor, shape=[n_edges, n_faces]
            Cochain operator C^2 -> C^1.

        Returns
        -------
        zv_out : torch.tensor, shape=[n_nodes, target_ch_v]
            Output feature vector on the nodes of the input SC/CX.
        ze_out : torch.tensor, shape=[n_edges, target_ch_e]
            Output feature vector on the edges of the input SC/CX.
        zf_out : torch.tensor, shape=[ n_edges, target_ch_f]
            Output feature vector on the faces of the input SC/CX.
        """
        if xv is not None:
            xv = F.dropout(xv, self.dropout, self.training)
        if xe is not None:
            xe = F.dropout(xe, self.dropout, self.training)
        if xf is not None:
            xf = F.dropout(xf, self.dropout, self.training)

        if Ge2v is None:
            v2e = None
        else:
            v2e = torch.t(Ge2v)

        if Gf2e is None:
            e2f = None
        else:
            e2f = torch.t(Gf2e)

        ze_out = self.merge_v_f(xv, xf, v2e, Gf2e)  # target:edges

        zv_out, zf_out = self.split_v_f(xe, Ge2v, e2f)  # targets : (vertices , faces)

        if self.act is not None:
            zv_out = self.act(zv_out)
            ze_out = self.act(ze_out)
            zf_out = self.act(zf_out)

        return zv_out, ze_out, zf_out
