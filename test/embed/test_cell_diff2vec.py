"""Test CellDiff2Vec class."""

import unittest

import toponetx as tnx

from topomodelx.embed.cell_diff2vec import CellDiff2Vec


class TestDiff2Vec(unittest.TestCase):
    """Test Diff2Vec class."""

    def test_init(self):
        """Test get_embedding."""
        # Create a small graph
        sc = tnx.SimplicialComplex()
        sc.add_simplex([0, 1])

        # Create a CellDiff2Vec object
        _ = CellDiff2Vec(dimensions=2)


if __name__ == "__main__":
    unittest.main()
