"""Convert complexes to simplicial complexes."""

import gudhi
from toponetx import SimplicialComplex


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


def rips_complex_point_cloud(point_cloud, max_edge_length, max_dimension=2):
    """Compute the Rips complex of a point cloud.

    Parameters
    ----------
    point_cloud : np.array
        Point cloud.
    max_edge_length : float
        Maximum length of an edge.
    max_dimension : int, optional
        Maximum dimension. The default is 2.

    Returns
    -------
    _ : SimplicialComplex
        Rips complex.
    """
    rips_complex = gudhi.RipsComplex(
        points=point_cloud, max_edge_length=max_edge_length
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    faces_dict = SimplicialComplex.extract_simplices(simplex_tree)
    lst = []
    for i in range(0, len(faces_dict)):
        lst = lst + list(faces_dict[i].keys())
    return SimplicialComplex(lst)


def rips_complex_distance_matrix(distance_mat, max_edge_length, max_dimension=2):
    """Compute the Rips complex from distance matrix.

    Parameters
    ----------
    distance_mat : np.array
        Distance matrix.
    max_edge_length : float
        Maximum length of an edge.
    max_dimension : int, optional
        Maximum dimension. The default is 2.

    Returns
    -------
    _ : SimplicialComplex
        Rips complex.
    """
    rips_complex = gudhi.RipsComplex(
        distance_matrix=distance_mat, max_edge_length=max_edge_length
    )
    simplex_tree = rips_complex.create_simplex_tree(max_dimension=max_dimension)

    faces_dict = SimplicialComplex.extract_simplices(simplex_tree)
    lst = []
    for i in range(0, len(faces_dict)):
        lst = lst + list(faces_dict[i].keys())
    return SimplicialComplex(lst)
