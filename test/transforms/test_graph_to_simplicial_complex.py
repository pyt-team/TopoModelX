"""Test graph to simplicial complex transforms."""

import unittest

import networkx as nx

from topomodelx.transforms import (
    get_all_clique_complex_incidence_matrices,
    get_k_cliques,
)


class TestGraphToSimplicialComplex(unittest.TestCase):
    """Test graph to simplicial complex transforms."""

    def test_get_k_cliques(self):
        """Test get_k_cliques."""
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

    def test_get_graph_incidence_matrices(self):
        """Test get_graph_incidence_matrices."""
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
        print(B0.shape)
        print(len(zero_cliques))
        assert B0.shape[0] == len(zero_cliques), B0.shape
        assert B0.shape[1] == len(one_cliques), B0.shape

        assert B1.shape[0] == len(one_cliques), B1.shape
        assert B1.shape[1] == len(two_cliques), B1.shape

        assert B2.shape[0] == len(two_cliques), B2.shape
        assert B2.shape[1] == len(three_cliques), B2.shape

        assert B3.shape[0] == len(three_cliques), B3.shape
        assert B3.shape[1] == len(four_cliques), B3.shape


if __name__ == "__main__":
    unittest.main()
