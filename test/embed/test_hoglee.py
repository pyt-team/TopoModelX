"""Test the HOGLEE class."""

import unittest

import numpy as np
import toponetx as tnx

from topomodelx.embed.hoglee import HOGLEE


class TestHOGLEE(unittest.TestCase):
    """Test the HOGLEE class."""

    def test_fit_and_get_embedding(self):
        """Test get_embedding."""
        # Create a small complex
        cx = tnx.CellComplex([[1, 2, 3, 4], [3, 4, 5, 6, 7, 8]], ranks=2)

        # Create a Cell2Vec object
        dc = HOGLEE(dimensions=5)

        # Fit the Cell2Vec object to the graph and get embedding for nodes (using adjacency matrix A0)
        dc.fit(cx, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1})

        # Check that the shape of the embedding is correct
        assert dc.get_embedding().shape == (len(cx.nodes), 5 + 1)

        # Check that the shape of the embedding dictionary is correct
        ind = dc.get_embedding(get_dict=True)
        assert (len(ind)) == len(cx.nodes)

        # Check that the embedding of the first node is not equal to the embedding of the second node
        assert not np.allclose(dc.get_embedding()[0], dc.get_embedding()[1])


if __name__ == "__main__":
    unittest.main()
