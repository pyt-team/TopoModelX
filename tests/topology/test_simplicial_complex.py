import sys
import unittest

import networkx as nx
import numpy as np
from scipy.linalg import fractional_matrix_power
# from stnets.topology import SimplicialComplex
from sklearn.preprocessing import normalize

from torch_topo.topology import SimplicialComplex


def normalize_(A):  # works on numpy arrays --
    # A is assumed to be an adj matrix, without selfloops
    I = np.identity(A.shape[0])
    A_hat = A + I
    D = np.diag(np.sum(A_hat, axis=0))
    D_half_norm = fractional_matrix_power(D, -0.5)
    return D_half_norm.dot(A_hat).dot(D_half_norm)


def test_simplical_complex():
    simplices = [(0, 1, 2), (1, 2, 3), (2, 3), (1, 2, 4), (5, 3), (0, 4)]
    HL = SimplicialComplex(simplices, mode="gudhi")

    B1 = HL.get_boundary_operator(1)
    B2 = HL.get_boundary_operator(2)

    L0 = HL.get_hodge_laplacian(0)
    L1 = HL.get_hodge_laplacian(1)
    L2 = HL.get_hodge_laplacian(2)

    N0 = len(HL.n_faces(0))
    N1 = len(HL.n_faces(1))
    N2 = len(HL.n_faces(2))

    assert HL.get_dimension() == 2

    assert N0 == 6
    assert N1 == 9
    assert N2 == 3

    assert L0.shape == (6, 6)
    assert L1.shape == (9, 9)
    assert L2.shape == (3, 3)

    HL = SimplicialComplex(simplices, maxdimension=1, mode="gudhi")

    assert HL.get_dimension() == 1

    Adj0 = HL.get_higher_order_adj(0)
    D1 = np.sort(Adj0.toarray().sum(axis=1))
    E = [list(i) for i in HL.get_simplices(1)]
    G1 = nx.Graph()
    G1.add_edges_from(sorted(E))

    D2 = np.sort(nx.to_scipy_sparse_array(G1).toarray().sum(axis=1))

    def get_edges_from_operator(adj):
        rows, cols = np.where(np.sign(np.abs(adj.toarray())) == 1)
        edges = zip(rows.tolist(), cols.tolist())
        return edges

    edges2 = get_edges_from_operator(Adj0)
    G2 = nx.Graph()
    G2.add_edges_from(edges2)
    val = nx.is_isomorphic(G1, G2)

    assert D1.all() == D2.all()
    assert val == True

    HL = SimplicialComplex(simplices, mode="gudhi")

    Adj1 = HL.get_higher_order_adj(1)

    Adj1N_kipf = HL.get_normalized_higher_order_adj(1, normalization="kipf").toarray()
    Adj1N_xu = HL.get_normalized_higher_order_adj(1, normalization="xu").toarray()
    Adj1N_row = HL.get_normalized_higher_order_adj(1, normalization="row").toarray()

    Adj1N_kipf_d = HL.asymmetric_kipf_normalization(
        Adj1.toarray() + np.eye(Adj1.shape[0]), is_sparse=False
    )
    Adj1N_xu_d = HL.asymmetric_xu_normalization(
        Adj1.toarray() + np.eye(Adj1.shape[0]), is_sparse=False
    )
    Adj1N_row_d = normalize(Adj1.toarray() + np.eye(Adj1.shape[0]), norm="l1", axis=1)

    # normalization can be computed for dense and sparse matrices

    assert Adj1N_kipf.all() == Adj1N_kipf_d.all()
    assert Adj1N_xu.all() == Adj1N_xu_d.all()
    assert Adj1N_row.all() == Adj1N_row_d.all()

    # normalization does not change the sign of the boundary maps
    assert (
        np.sign(
            HL.get_normalized_boundary_operator(1, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(1).toarray().any()
    )

    assert (
        np.sign(
            HL.get_normalized_boundary_operator(2, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(2).toarray().any()
    )
    # normalization does not change the sign of the coboundary maps
    assert (
        np.sign(
            HL.get_normalized_coboundary_operator(1, normalization="row").toarray()
        ).any()
        == HL.get_coboundary_operator(1).toarray().any()
    )

    assert (
        np.sign(
            HL.get_normalized_coboundary_operator(2, normalization="row").toarray()
        ).any()
        == HL.get_coboundary_operator(2).toarray().any()
    )


def large_simplical_complex():
    # test for larger example
    G = nx.karate_club_graph()
    cliques = list(nx.enumerate_all_cliques(G))

    HL = SimplicialComplex(cliques)

    assert (
        np.sign(
            HL.get_normalized_boundary_operator(1, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(1).toarray().any()
    )

    assert (
        np.sign(
            HL.get_normalized_boundary_operator(2, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(2).toarray().any()
    )

    assert (
        np.sign(
            HL.get_normalized_boundary_operator(3, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(3).toarray().any()
    )

    assert (
        np.sign(
            HL.get_normalized_boundary_operator(4, normalization="row").toarray()
        ).any()
        == HL.get_boundary_operator(4).toarray().any()
    )


if __name__ == "__main__":
    unittest.main()
