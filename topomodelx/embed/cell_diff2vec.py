"""Class CellDiff2Vec."""

import networkx as nx
from karateclub import Diff2Vec

from topoembedx.neighborhood import neighborhood_from_complex


class CellDiff2Vec(Diff2Vec):
    """Class for CellDiff2Vec.

    Parameters
    ----------
    diffusion_number : int, optional
        Number of diffusion. Defaults to 10.
    diffusion_cover : int, optional
        Number of nodes in diffusion. Defaults to 80.
    dimensions : int, optional
        Dimensionality of embedding. Defaults to 128.
    workers : int, optional
        Number of cores. Defaults to 4.
    window_size : int, optional
        Matrix power order. Defaults to 5.
    epochs : int, optional
        Number of epochs. Defaults to 1.
    learning_rate : float, optional
        HogWild! learning rate. Defaults to 0.05.
    min_count : int, optional
        Minimal count of node occurrences. Defaults to 1.
    seed : int, optional
        Random seed value. Defaults to 42.
    """

    def __init__(
        self,
        diffusion_number: int = 10,
        diffusion_cover: int = 80,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):
        super().__init__(
            diffusion_number=diffusion_number,
            diffusion_cover=diffusion_cover,
            dimensions=dimensions,
            workers=workers,
            window_size=window_size,
            epochs=epochs,
            learning_rate=learning_rate,
            min_count=min_count,
            seed=seed,
        )

        self.A = []
        self.ind = []

    def fit(self, complex, neighborhood_type="adj", neighborhood_dim={"r": 0, "k": -1}):
        """Fit a CellDiff2Vec model.

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

        print("HELO")
        print(g)
        print(self.A)
        super(CellDiff2Vec, self).fit(g)

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
        emb = super(CellDiff2Vec, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        else:
            return emb
