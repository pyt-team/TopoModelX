import unittest

import networkx as nx

from torch_topo.topology import SimplicialComplex
from torch_topo.transforms import (
    get_all_clique_complex_incidence_matrices,
    get_k_cliques,
)


def test_clique_complex():
    G = nx.karate_club_graph()
    cliques = list(nx.enumerate_all_cliques(G))
    nodes = [i for i in cliques if len(i) == 1]
    edges = [i for i in cliques if len(i) == 2]
    faces = [i for i in cliques if len(i) == 3]
    threefaces = [i for i in cliques if len(i) == 4]
    fourfaces = [i for i in cliques if len(i) == 5]

    SC = SimplicialComplex(cliques)
    assert len(SC.get_simplices(0)) == len(nodes)
    assert len(SC.get_simplices(1)) == len(edges)
    assert len(SC.get_simplices(2)) == len(faces)
    assert len(SC.get_simplices(3)) == len(threefaces)
    assert len(SC.get_simplices(4)) == len(fourfaces)


def test_get_k_cliques():
    G = nx.karate_club_graph()
    cliques = list(nx.enumerate_all_cliques(G))
    zero_cliques = list(get_k_cliques(G, 1))
    one_cliques = list(get_k_cliques(G, 2))
    two_cliques = list(get_k_cliques(G, 3))
    three_cliques = list(get_k_cliques(G, 4))
    four_cliques = list(get_k_cliques(G, 5))

    assert len([c for c in cliques if len(c) == 1]) == len(zero_cliques)
    assert len([c for c in cliques if len(c) == 2]) == len(one_cliques)
    assert len([c for c in cliques if len(c) == 3]) == len(two_cliques)
    assert len([c for c in cliques if len(c) == 4]), len(three_cliques)
    assert len([c for c in cliques if len(c) == 5]), len(four_cliques)


def test_get_graph_incidence_matrices():
    G = nx.karate_club_graph()

    zero_cliques = list(get_k_cliques(G, 1))
    one_cliques = list(get_k_cliques(G, 2))
    two_cliques = list(get_k_cliques(G, 3))
    three_cliques = list(get_k_cliques(G, 4))
    four_cliques = list(get_k_cliques(G, 5))

    matrices = get_all_clique_complex_incidence_matrices(G)
    B0 = matrices[0]
    B1 = matrices[1]
    B2 = matrices[2]
    B3 = matrices[3]
    B4 = matrices[4]
    assert B0.shape[-1] == len(zero_cliques)

    assert B1.shape[0] == len(zero_cliques)
    assert B1.shape[1] == len(one_cliques)

    assert B2.shape[0] == len(one_cliques)
    assert B2.shape[1] == len(two_cliques)

    assert B3.shape[0] == len(two_cliques)
    assert B3.shape[1] == len(three_cliques)

    assert B4.shape[0] == len(three_cliques)
    assert B4.shape[1] == len(four_cliques)


if __name__ == "__main__":
    unittest.main()
