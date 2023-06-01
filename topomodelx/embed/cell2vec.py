"""Cell2Vec: a class that extends the Node2Vec class."""

import networkx as nx
from karateclub import Node2Vec

from topoembedx.neighborhood import neighborhood_from_complex


class Cell2Vec(Node2Vec):
    """Class Cell2Vec.

    Cell2Vec is a class that extends the Node2Vec class
    and provides additional functionality for generating node embeddings for simplicial, cell, combinatorial,
    or dynamic combinatorial complexes. The Cell2Vec class takes as input a simplicial, cell, combinatorial,
    or dynamic combinatorial complex, and uses the adjacency matrix or coadjacency matrix of the complex to
    create a graph object using the networkx library. The Cell2Vec class then uses this graph object to generate
    node embeddings using the Node2Vec algorithm. The Cell2Vec class allows users to specify the type of adjacency
    or coadjacency matrix to use for the graph (e.g. "adj" for adjacency matrix or "coadj" for coadjacency matrix),
    as well as the dimensions of the neighborhood to use for the matrix (e.g. the "r" and "k" values for the matrix).
    Additionally, users can specify the dimensions of the node embeddings to generate, the length and number of
    random walks to use for the node2vec algorithm, and the number of workers to use for parallelization.


    Parameters
    ----------
    walk_number : int, optional
        Number of random walks to start at each node. Defaults to 10.
    walk_length : int, optional
        Length of random walks. Defaults to 80.
    p : float, optional
        Return parameter (1/p transition probability) to move towards from previous node. Defaults to 1.0.
    q : float, optional
        In-out parameter (1/q transition probability) to move away from previous node. Defaults to 1.0.
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
        walk_number: int = 10,
        walk_length: int = 80,
        p: float = 1.0,
        q: float = 1.0,
        dimensions: int = 128,
        workers: int = 4,
        window_size: int = 5,
        epochs: int = 1,
        learning_rate: float = 0.05,
        min_count: int = 1,
        seed: int = 42,
    ):

        super().__init__(
            walk_number=walk_number,
            walk_length=walk_length,
            p=p,
            q=q,
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
        """Fit a Cell2Vec model.

        Parameters
        ----------
        complex : TopoNetX object
            A complex object.
            Can be one of the following:
            - CellComplex
            - CombinatorialComplex
            - CombinatorialComplex
            - SimplicialComplex
            - DynamicSimplicialComplex
        neighborhood_type : str
            The type of neighborhood to compute.
            Can be one of the following:
             - "adj" for adjacency matrix.
             - "coadj" for coadjacency matrix.
        neighborhood_dim : dict
            The dimensions of the neighborhood to use.
            If `neighborhood_type` is "adj", dimension is `neighborhood_dim['r']`.
            If `neighborhood_type` is "coadj", dimension is `neighborhood_dim['k']`
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

        super(Cell2Vec, self).fit(g)

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
        emb = super(Cell2Vec, self).get_embedding()
        if get_dict:
            return dict(zip(self.ind, emb))
        else:
            return emb
