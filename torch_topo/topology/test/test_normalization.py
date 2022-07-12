import sys
import unittest

import networkx as nx
import numpy as np

sys.path.append("..")

# from simplicial_complex import SimplicialComplex

from normalization import (
    _compute_B1_normalized,
    _compute_B1T_normalized,
    _compute_B2_normalized,
    _compute_B2T_normalized,
    _compute_D1,
    _compute_D2,
    _compute_D3,
    _compute_D5,
    get_normalized_2d_operators,
)

from stnets.topology import SimplicialComplex


class test_normalization(unittest.TestCase):
    def test_compute_B1_normalized(self):
        G = nx.karate_club_graph()
        cliques = list(nx.enumerate_all_cliques(G))

        SC = SimplicialComplex(cliques)
        B1 = SC.get_boundary_operator(1)
        B2 = SC.get_boundary_operator(2)
        B1N, B1TN, B2N, B2TN = get_normalized_2d_operators(B1, B2)
        self.assertAlmostEqual(B1.shape, B1N.shape)
        self.assertAlmostEqual(B2.shape, B2N.shape)
        self.assertAlmostEqual(B1TN.shape, B1.T.shape)
        self.assertAlmostEqual(B2.T.shape, B2TN.shape)
        self.assertAlmostEqual(
            _compute_B1_normalized(B1.toarray(), B2.toarray()).all(),
            _compute_B1_normalized(B1, B2).toarray().all(),
        )

        self.assertAlmostEqual(
            _compute_B1T_normalized(B1.toarray(), B2.toarray()).all(),
            _compute_B1T_normalized(B1, B2).toarray().all(),
        )

        self.assertAlmostEqual(
            _compute_B2_normalized(B2.toarray()).all(),
            _compute_B2_normalized(B2).toarray().all(),
        )

        self.assertAlmostEqual(
            _compute_B2T_normalized(B2.toarray()).all(),
            _compute_B2T_normalized(B2).toarray().all(),
        )

        self.assertAlmostEqual(np.sign(B1N.toarray()).all(), B1.toarray().all())
        self.assertAlmostEqual(np.sign(B1TN.toarray()).all(), B1.T.toarray().all())
        self.assertAlmostEqual(np.sign(B2N.toarray()).all(), B2.toarray().all())
        self.assertAlmostEqual(np.sign(B2TN.toarray()).all(), B2.T.toarray().all())


if __name__ == "__main__":
    unittest.main()
