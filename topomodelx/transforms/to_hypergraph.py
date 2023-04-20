__all__ = [
    "simplicial_subcomplex_2_hypergraph_incidence_matrix",
    "simplicial_complex_2_hypergraph",
    "graph_2_neighborhood_hypergraph",
    "pointcloud_2_knn_graph",
    "graph_2_k_hop_hypergraph",
    "graph_2_k_hop_hypergraph",
    "simplicial_complex_closure_of_hypergraph",
    "distance_matrix_2_knn_hypergraph",
    "distance_matrix_2_knn_graph",
    "distance_matrix_2_eps_neighborhood_hypergraph",
]

import hypernetx as hnx
import networkx as nx
import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import kneighbors_graph, radius_neighbors_graph
from toponetx.classes.simplicial_complex import SimplicialComplex as SC


def simplicial_subcomplex_2_hypergraph_incidence_matrix(
    simplicial_complex, nodes_dim, edges_dim
):
    """
    Parameters
    ----------
    simplicial_complex : SimplicialComplex object
        DESCRIPTION. a simplicial complex object
    nodes_dim : int
        DESCRIPTION. represent the dimension of the simplicies in the simplices complex that we
                     to consider as being the the nodes of the output hypergraph
    nodes_dim : int
        DESCRIPTION. represent the dimension of the simplicies in the simplices complex that we
                     to consider as being the the edges of the output hypergraph
    Returns
    -------
    TYPE
        an indicence matrix of a hypergraph whose nodes are
        simplices of dimension nodes_dim and whose edges are simplices of dimension edges_dim
        a node i is a part of the edge j if the intersection of i and j is not empty.

    """
    assert isinstance(nodes_dim, int)
    assert isinstance(edges_dim, int)
    assert isinstance(simplicial_complex, SimplicialComplex)
    max_dim = simplicial_complex.maxdim
    assert nodes_dim <= max_dim
    assert edges_dim <= max_dim
    _nodes = simplicial_complex.dic_order_faces(
        nodes_dim
    )  # maintain the same order stored in the SC
    _edges = simplicial_complex.dic_order_faces(
        edges_dim
    )  # maintain the same order stored in the SC
    graph = []
    for i in range(0, len(_edges)):
        edge = []
        for j in range(0, len(_nodes)):
            if _nodes[j] & _edges[i]:
                edge.append(1)
            else:
                edge.append(0)
        graph.append(edge)
    return np.array(graph).T


def simplicial_complex_closure_of_hypergraph(H):
    """
    Parameters
    ----------
    H : hyernetx hypergraph
        DESCRIPTION. hypergraph

    Returns
    -------
    TYPE
        simplicial complex closure of the hypergraph

    """
    edges = H.edges
    lst = []
    for e in edges:
        lst.append(edges[e])
    return SimplicialComplex(lst)


def simplicial_complex_2_hypergraph(simplicial_complex):
    """
    Parameters
    ----------
    simplicial_complex : SimplicialComplex object
        DESCRIPTION. a simplicial complex object
    Returns
    -------
    TYPE hypernetx hypergraph
        a hypergraph whose edges are all sets in the simplicial complex
        that have cardinalties larger than or equal to 2.

    """
    assert isinstance(simplicial_complex, SimplicialComplex)
    max_dim = simplicial_complex.maxdim

    graph = []
    for i in range(1, max_dim + 1):
        edge = [list(j) for j in simplicial_complex.n_faces(i)]
        graph = graph + edge
    return hnx.Hypergraph(graph, static=True)


def graph_2_neighborhood_hypergraph(G):
    """
    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    TYPE : hypernetx hypergraph

    """
    edges = [sorted(list(G.neighbors(v)) + [v]) for v in G.nodes]

    return hnx.Hypergraph(edges, static=True)


def graph_2_k_hop_hypergraph(G, k_hop=1):
    """
    Parameters
    ----------
    G : networkx graph

    Returns
    -------
    TYPE : hypernetx hypergraph

    """
    edges = [sorted(list(nx.ego_graph(G, v, k_hop).nodes())) for v in G.nodes]

    return hnx.Hypergraph(edges, static=True)


def pointcloud_2_knn_graph(pointcloud, num_neighbord):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. a collection of Euclidean points in R^n
    num_neighbord : TYPE
        DESCRIPTION.

    Returns
    -------
    G : networkx graph
        DESCRIPTION. the knn weighted graph obtained from the point cloud
                     weight is the distance between the points.
    """
    A = kneighbors_graph(pointcloud, num_neighbord, mode="distance")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return G


def pointcloud_2_knn_hypergraph(pointcloud, num_neighbord):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. a collection of Euclidean points in R^n
    num_neighbord : TYPE
        DESCRIPTION.

    Returns
    -------
    G : hypenetx hypergraph
        DESCRIPTION.
    """
    A = kneighbors_graph(pointcloud, num_neighbord, mode="distance")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def pointcloud_2_eps_neighborhood_hypergraph(pointcloud, eps):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. a collection of Euclidean points in R^n
    eps : float,
        DESCRIPTION. real nymber representing the thinkness around each point
                     in the metric space

    Returns
    -------
    G : hypenetx hypergraph
        DESCRIPTION.
    """
    A = radius_neighbors_graph(pointcloud, eps, mode="connectivity")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def distance_matrix_2_eps_neighborhood_hypergraph(distance_matrix, eps):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. sparse coo_matrix distance matrix
    eps : float,
        DESCRIPTION. real nymber representing the thinkness around each point
                     in the metric space

    Returns
    -------
    G : hypenetx hypergraph
        DESCRIPTION.
    """
    G = nx.Graph()
    for i, j, v in zip(distance_matrix.row, distance_matrix.col, distance_matrix.data):
        if v <= eps:
            G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def distance_matrix_2_knn_graph(distance_matrix, num_neighbords):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. sparse coo_matrix distance matrix
    eps : float,
        DESCRIPTION. real nymber representing the thinkness around each point
                     in the metric space

    Returns
    -------
    G : hypenetx hypergraph
        DESCRIPTION.
    """

    knn = kneighbors_graph(distance_matrix, num_neighbords, metric="precomputed")
    G = nx.Graph()
    cx = coo_matrix(knn)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return G


def distance_matrix_2_knn_hypergraph(distance_matrix, num_neighbords):

    """
    Parameters
    ----------
    pointcloud : numpy array
        DESCRIPTION. sparse coo_matrix distance matrix
    eps : float,
        DESCRIPTION. real nymber representing the thinkness around each point
                     in the metric space

    Returns
    -------
    G : hypenetx hypergraph
        DESCRIPTION.
    """
    G = distance_matrix_2_knn_graph(distance_matrix, num_neighbords)
    return graph_2_neighborhood_hypergraph(G)
