"""Convert a graph to a simplicial complex."""

__all__ = [
    "get_k_cliques",
    "graph_clique_complex",
    "get_all_clique_complex_incidence_matrices",
    "get_clique_complex_incidence_matrix",
    "create_higher_order_adj_from_edges",
    "get_neighbor_complex",
]


import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from toponetx import SimplicialComplex


def get_neighbor_complex(G):
    """Get the neighbor complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.

    Returns
    -------
    _ : SimplicialComplex
        The neighbor complex of the graph.

    Notes
    -----
    This type of simplicial complexes can have very large dimension
    and it is a function of the distribution of the valency of the graph.

    """
    neighbors = []
    for i in G.nodes():
        N = list(G.neighbors(i)) + [i]
        neighbors.append(N)
    return SimplicialComplex(neighbors)


def get_all_neighbor_complex_incidence_matrices(G, max_dim=None, signed=True):
    """Get all incidence matrices.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    dim : integer, optional
        The max dimension of the cliques in the output clique complex.
        The default is None indicate max dimension.
    signed : bool, optional
        indicates if the output matrix is signed or not.
        The default is True.

    Returns
    -------
    _ : list
        List of incidence matrices B1,..B_{k} where
        k=min(dim,max_dim_of_complex).
    """
    complex = get_neighbor_complex(G, max_dim)
    return [complex.incidence_matrix(i, signed) for i in range(0, complex.dim + 1)]


def get_k_cliques(G, k):
    """Get cliques of dimension k.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    k : int
        The dimenion of the clique we want to extract from the graph.

    Returns
    -------
    _ :
        Generator for all k cliques in the graph.
    """
    return filter(lambda face: len(face) == k, nx.enumerate_all_cliques(G))


def graph_clique_complex(G, dim=None):
    """Get the clique complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    dim : int, optional
        The max dimension of the cliques in
        the output clique complex.
        The default is None indicate max dimension.

    Returns
    -------
    _ : SimplicialComplex
        The clique simplicial complex of dimension dim of the graph G.
    """
    if dim is None:
        lst = nx.enumerate_all_cliques(G)
        return SimplicialComplex(list(lst))

    lst = filter(lambda face: len(face) <= dim, nx.enumerate_all_cliques(G))
    return SimplicialComplex(list(lst))


def get_all_clique_complex_incidence_matrices(G, max_dim=None, signed=True):
    """Get all incidence matrices of the clique complex of a graph.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    dim : int, optional
        The max dimension of the cliques in the output clique complex.
        The default is None indicate max dimension.
    signed : bool, optional
        Indicates if the output matrix is signed or not.
        The default is True.

    Returns
    -------
    _ : list
        List of incidence matrices B1,..B_{k} where
        k=min(dim,max_dim_of_complex).
    """
    complex = graph_clique_complex(G, max_dim)
    return [complex.incidence_matrix(i, signed) for i in range(1, complex.dim + 1)]


def get_clique_complex_incidence_matrix(G, dim, signed=True):
    """Get the incidence matrix of the clique complex of dimension dim of the graph G.

    Parameters
    ----------
    G : networkx graph
        Input graph.
    dim : int
        The dimension of the output incidence matrix
    signed : bool, optional
        Indicates if the output matrix is signed or not.
        The default is True.

    Returns
    -------
    _ :
        Incidence matrix B_dim.
    """
    complex = graph_clique_complex(G, dim + 1)
    return complex.incidence_matrix(dim, signed)


def create_higher_order_adj_from_edges(edges, shape):
    """Create a higher order adjacency matrix from a list of edges.

    Parameters
    ----------
    edges : np.array
        Edges.
    shape : tuple
        Shape of the output adjacency matrix.

    Returns
    -------
    _ : np.array
        Higher order adjacency matrix.
    """
    adj = coo_matrix((np.ones(edges.shape[0]), edges.T), shape=shape, dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
