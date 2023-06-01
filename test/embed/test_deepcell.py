"""Test the DeepCell class."""

import unittest

import numpy as np
import toponetx as tnx

from topomodelx.embed.deepcell import DeepCell


class TestDeepCell(unittest.TestCase):
    """Test the DeepCell class."""

    def test_fit_and_get_embedding(self):
        """Test fit and get_embedding."""
        # Create a small graph
        cx = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a DeepCell object
        dc = DeepCell(walk_number=5, walk_length=10, dimensions=2)

        # Fit the DeepCell object to the graph and get embedding for edges (using adjacency matrix A1)
        dc.fit(cx, neighborhood_type="adj", neighborhood_dim={"r": 1, "k": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(cx.edges), 2)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dict=True)
        assert (len(ind)) == len(cx.edges)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])


if __name__ == "__main__":
    unittest.main()
