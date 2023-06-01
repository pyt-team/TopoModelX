"""Higher Order Geometric Laplacian EigenMaps (HOGLEE) class."""
import networkx as nx
import numpy as np
from karateclub import GLEE

from topoembedx.neighborhood import neighborhood_from_complex


class HOGLEE(GLEE):
    """Class for Higher Order Geometric Laplacian EigenMaps (HOGLEE).

    Parameters
    ----------
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 3.
    seed : int, optional
        Random seed value. Defaults to 42.
    """

    def __init__(self, dimensions: int = 3, seed: int = 42):

        super().__init__(dimensions=dimensions, seed=seed)

        self.A = []
        self.ind = []

    def fit(self, complex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        """Fit a Higher Order Geometric Laplacian EigenMaps model.

        Parameters
        ----------
        complex : TopoNetX object
            A complex object. The complex object can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - CombinatorialComplex
            - SimplicialComplex
            - DynamicSimplicialComplex
        neighborhood_type : str
            The type of neighborhood to compute. "adj" for adjacency matrix, "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The dimensions of the neighborhood to use. If `neighborhood_type` is "adj", the dimension is
            `neighborhood_dim['r']`. If `neighborhood_type` is "coadj", the dimension is `neighborhood_dim['k']`
            and `neighborhood_dim['r']` specifies the dimension of the ambient space.

        Notes
        -----
        Here, neighborhood_dim={"r": 1, "k": -1} specifies the dimension for
        which the cell embeddings are going to be computed.
        r=1 means that the embeddings will be computed for the first dimension.
        The integer 'k' is ignored and only considered
        when the input complex is a combinatorial complex.

        Returns
        -------
        None
        """
        self.ind, self.A = neighborhood_from_complex(
            complex, neighborhood_type, neighborhood_dim
        )

        g = nx.from_numpy_matrix(self.A)

        super(HOGLEE, self).fit(g)

    def get_embedding(self, get_dict=False):
        """Get embedding.

        Parameters
        ----------
        get_dict : bool, optional
            Whether to return a dictionary. Defaults to False.

        Returns
        -------
        dict or numpy.ndarray
            Embedding.
        """
        emb = super(HOGLEE, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        else:
            return emb
