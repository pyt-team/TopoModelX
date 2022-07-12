__all__ = [
    "get_k_cliques",
    "graph_clique_complex",
    "get_all_clique_complex_incidence_matrices",
    "get_clique_complex_incidence_matrix",
    "create_higher_order_adj_from_edges",
    "get_neighbor_complex",
]


from itertools import combinations

import networkx as nx
import numpy as np
from numpy.linalg import inv, pinv
from scipy.sparse import coo_matrix

from stnets.topology import simplicial_complex as SC


def get_neighbor_complex(G):
    """
    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    TYPE simplical complex object

    Note:
    -----
        This type of simplicial complexes can have very large dimension
        and it is a function of the distribution of the valency of the graph.

    """
    neighbors = []
    for i in G.nodes():
        N = list(G.neighbors(i)) + [i]
        neighbors.append(N)
    return SC.SimplicialComplex(neighbors)


def get_all_neighbor_complex_incidence_matrices(G, max_dim=None, signed=True):
    """
    Args
    ----------
        G : networkx graph
        dim : integer, optional
            The max dimension of the cliques in the output clique complex.
            The default is None indicate max dimension.
        signed : Bool, optional
            indicates if the output matrix is signed or not.
            The default is True.
    Returns
    -------
        list of incidence matrices B1,..B_{k} where
        k=min(dim,max_dim_of_complex).
    """
    cplex = get_neighbor_complex(G, max_dim)
    return [
        cplex.get_boundary_operator(i, signed)
        for i in range(0, cplex.get_dimension() + 1)
    ]


def get_k_cliques(G, k):
    """
    Args
    -----
        G : networkx graph
        k : integer
            The dimenion of the clique we want to extract from the graph.
    Returns
    -------
         a genrator for all k cliques in the graph.
    """
    return filter(lambda face: len(face) == k, nx.enumerate_all_cliques(G))


def graph_clique_complex(G, dim=None):
    """
    Args
    ----------
        G : networkx graph
        dim : integer, optional
            The max dimension of the cliques in
            the output clique complex.
            The default is None indicate max dimension.
    Returns
    -------
        The clique simplicial complex of dimension dim of the graph G
    """
    if dim is None:
        lst = nx.enumerate_all_cliques(G)
        return SC.SimplicialComplex(list(lst))
    else:
        lst = filter(lambda face: len(face) <= dim, nx.enumerate_all_cliques(G))
        return SC.SimplicialComplex(list(lst))


def get_all_clique_complex_incidence_matrices(G, max_dim=None, signed=True):
    """
    Args
    ----------
        G : networkx graph
        dim : integer, optional
            The max dimension of the cliques in the output clique complex.
            The default is None indicate max dimension.
        signed : Bool, optional
            indicates if the output matrix is signed or not.
            The default is True.
    Returns
    -------
        list of incidence matrices B1,..B_{k} where
        k=min(dim,max_dim_of_complex).
    """
    cplex = graph_clique_complex(G, max_dim)
    return [
        cplex.get_boundary_operator(i, signed)
        for i in range(0, cplex.get_dimension() + 1)
    ]


def get_clique_complex_incidence_matrix(G, dim, signed=True):
    """
    Args
    ----------
        G : networkx graph
        dim : integer
            The dimension of the output incidence matrix
        signed : Bool, optional
            indicates if the output matrix is signed or not.
            The default is True.
    Returns
    -------
        incidence matrices B_dim
    """

    cplex = graph_clique_complex(G, dim + 1)
    return cplex.get_boundary_operator(dim, signed)


def create_higher_order_adj_from_edges(edges, shape):
    adj = coo_matrix((np.ones(edges.shape[0]), edges.T), shape=shape, dtype=np.float32)
    return adj + adj.T.multiply(adj.T > adj) - adj.multiply(adj.T > adj)
