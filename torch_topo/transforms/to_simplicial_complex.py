
from stnets.topology import SimplicialComplex as SC
import gudhi

def simplicial_complex_closure_of_hypergraph(H):
    edges = H.edges
    lst = []
    for e in edges:
        lst.append(edges[e])
    return SC(lst)    

def rips_complex_point_cloud(point_cloud, max_edge_length, max_dimension = 2 ):
    
    rips_complex=gudhi.RipsComplex(points = point_cloud,
                                 max_edge_length = max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = max_dimension)

    faces_dict = SC.extract_simplices(simplex_tree)
    lst = []
    for i in range(0, len(faces_dict)):
        lst = lst + list(faces_dict[i].keys())
    return SC(lst)    



def rips_complex_distance_matrix(distance_mat, max_edge_length, max_dimension = 2 ):
    
    rips_complex=gudhi.RipsComplex(distance_matrix = distance_mat,
                                 max_edge_length = max_edge_length)
    simplex_tree = rips_complex.create_simplex_tree(max_dimension = max_dimension)

    faces_dict = SC.extract_simplices(simplex_tree)
    lst = []
    for i in range(0, len(faces_dict)):
        lst = lst + list(faces_dict[i].keys())
    return SC(lst)            