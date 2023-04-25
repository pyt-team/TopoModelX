"""Convert complexes to hypergraphs."""

__all__ = [
    "simplicial_subcomplex_2_hypergraph_incidence_matrix",
    "simplicial_complex_2_hypergraph",
    "graph_2_neighborhood_hypergraph",
    "point_cloud_2_knn_graph",
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
from toponetx import SimplicialComplex


def simplicial_subcomplex_2_hypergraph_incidence_matrix(
    simplicial_complex, nodes_dim, edges_dim
):
    """Convert a simplicial subcomplex to a hypergraph incidence matrix.

    Parameters
    ----------
    simplicial_complex : SimplicialComplex
        Simplicial complex.
    nodes_dim : int
        Dimension of the simplicies in the simplicial complex that we
        consider as being the the nodes of the output hypergraph.
    edges_dim : int
        Dimension of the simplicies in the simplicial complex that we
        consider as being the the edges of the output hypergraph.

    Returns
    -------
    _ : np.array
        Indicence matrix of a hypergraph whose nodes are
        simplices of dimension nodes_dim and whose edges are simplices of dimension edges_dim.
        A node i is a part of the edge j if the intersection of i and j is not empty.
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
    """Compute the simplicial complex closure of a hypergraph.

    Parameters
    ----------
    H : hyernetx hypergraph
        Hypergraph.

    Returns
    -------
    _ : SimplicialComplex
        Simplicial complex closure of the hypergraph.
    """
    edges = H.edges
    lst = []
    for e in edges:
        lst.append(edges[e])
    return SimplicialComplex(lst)


def simplicial_complex_2_hypergraph(simplicial_complex):
    """Convert a simplicial complex to a hypergraph.

    Parameters
    ----------
    simplicial_complex : SimplicialComplex
        Simplicial complex.

    Returns
    -------
    _ : hypernetx hypergraph
        Hypergraph whose edges are all sets in the simplicial complex
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
    """Convert a graph to a hypergraph.

    Parameters
    ----------
    G : networkx graph
        Graph.

    Returns
    -------
    _ : hypernetx hypergraph
        Hypergraph.
    """
    edges = [sorted(list(G.neighbors(v)) + [v]) for v in G.nodes]

    return hnx.Hypergraph(edges, static=True)


def graph_2_k_hop_hypergraph(G, k_hop=1):
    """Convert a graph to a hypergraph.

    Parameters
    ----------
    G : networkx graph
        Graph.

    Returns
    -------
    _ : hypernetx hypergraph
        Hypergraph.

    """
    edges = [sorted(list(nx.ego_graph(G, v, k_hop).nodes())) for v in G.nodes]

    return hnx.Hypergraph(edges, static=True)


def point_cloud_2_knn_graph(point_cloud, n_neighbors):
    """Convert a point cloud to a knn graph.

    Parameters
    ----------
    point_cloud : np.array
        Collection of Euclidean points in R^n
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    G : networkx graph
        The knn weighted graph obtained from the point cloud.
        The weight is the distance between the points.
    """
    A = kneighbors_graph(point_cloud, n_neighbors, mode="distance")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return G


def point_cloud_2_knn_hypergraph(point_cloud, n_neighbors):
    """Convert a point cloud to a knn hypergraph.

    Parameters
    ----------
    point_cloud : np.array
        Collection of Euclidean points in R^n.
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    _ : hypenetx hypergraph
        Hypergraph.
    """
    A = kneighbors_graph(point_cloud, n_neighbors, mode="distance")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def point_cloud_2_eps_neighborhood_hypergraph(point_cloud, eps):
    """Convert a point cloud to a eps neighborhood hypergraph.

    Parameters
    ----------
    point_cloud : np.array
        Collection of Euclidean points in R^n.
    eps : float
        Real number representing the thinkness around each point
        in the metric space.

    Returns
    -------
    _ : hypenetx hypergraph
        Hypergraph.
    """
    A = radius_neighbors_graph(point_cloud, eps, mode="connectivity")
    G = nx.Graph()
    cx = coo_matrix(A)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def distance_matrix_2_eps_neighborhood_hypergraph(distance_matrix, eps):
    """Convert a distance matrix to a eps neighborhood hypergraph.

    Parameters
    ----------
    distance_matrix : array-like
        Sparse coo_matrix distance matrix.
    eps : float
        Real number representing the thinkness around each point
        in the metric space.

    Returns
    -------
    _ : hypenetx hypergraph
        Hypergraph.
    """
    G = nx.Graph()
    for i, j, v in zip(distance_matrix.row, distance_matrix.col, distance_matrix.data):
        if v <= eps:
            G.add_edge(i, j, weight=v)
    return graph_2_neighborhood_hypergraph(G)


def distance_matrix_2_knn_graph(distance_matrix, n_neighbors):
    """Convert a distance matrix to a knn graph.

    Parameters
    ----------
    distance_matrix : array-like
        Sparse coo_matrix distance matrix.
    n_neighbors : int
        Number of neighbors.

    Returns
    -------
    G : networkx graph
        Graph.
    """
    knn = kneighbors_graph(distance_matrix, n_neighbors, metric="precomputed")
    G = nx.Graph()
    cx = coo_matrix(knn)
    for i, j, v in zip(cx.row, cx.col, cx.data):
        G.add_edge(i, j, weight=v)
    return G


def distance_matrix_2_knn_hypergraph(distance_matrix, n_neighbors):
    """Convert a distance matrix to a knn hypergraph.

    Parameters
    ----------
    distance_matrix : array-like
        Sparse coo_matrix distance matrix.
    eps : float,
        real number representing the thinkness around each point
                     in the metric space

    Returns
    -------
    G : hypenetx hypergraph
        Hypergraph.
    """
    G = distance_matrix_2_knn_graph(distance_matrix, n_neighbors)
    return graph_2_neighborhood_hypergraph(G)
