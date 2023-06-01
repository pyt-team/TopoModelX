"""Higher Order Laplacian Eigenmaps."""

import networkx as nx
import numpy as np
from karateclub import LaplacianEigenmaps

from topoembedx.neighborhood import neighborhood_from_complex


class HigherOrderLaplacianEigenmaps(LaplacianEigenmaps):
    """Class for Higher Order Laplacian Eigenmaps.

    Parameters
    ----------
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 3.
    maximum_number_of_iterations : int, optional
        Maximum number of iterations. Defaults to 100.
    seed : int, optional
        Random seed value. Defaults to 42.
    """

    def __init__(
        self,
        dimensions: int = 3,
        maximum_number_of_iterations: int = 100,
        seed: int = 42,
    ):

        super().__init__(
            dimensions=dimensions,
            seed=seed,
        )

        self.A = []
        self.ind = []
        self.maximum_number_of_iterations = maximum_number_of_iterations

    def fit(self, complex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        """Fit a Higher Order Laplacian Eigenmaps model.

        Parameters
        ----------
        complex : SimplicialComplex, CellComplex, CombinatorialComplex, or CombinatorialComplex
            The complex to be embedded.
        neighborhood_type : str, optional
            The type of neighborhood to use, by default "adj".
        neighborhood_dim : dict, optional
            The dimension of the neighborhood to use, by default {"r": 0, "k": -1}.
        """
        self.ind, self.A = neighborhood_from_complex(
            complex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(HigherOrderLaplacianEigenmaps, self).fit(g)

    def get_embedding(self, get_dict=False):
        """Get embeddings.

        Parameters
        ----------
        get_dict : bool, optional
            Return a dictionary of the embedding, by default False

        Returns
        -------
        dict or np.ndarray
            The embedding of the complex.
        """
        emb = super(HigherOrderLaplacianEigenmaps, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        return emb
