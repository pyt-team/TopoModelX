"""Transform an input mesh to a simplicial complex."""

__all__ = ["read_mesh", "mesh_2_operators"]


import numpy as np
import torch
from scipy.sparse import coo_matrix, csr_matrix, diags, dok_matrix, eye
from toponetx import SimplicialComplex

from topomodelx.util.tensors_util import coo_2_torch_tensor


def read_mesh(path, file_type=".m"):
    """Read a mesh from a file.

    Parameters
    ----------
    path : string
       Path of a mesh.
    file_type: string
        Indicates the type of input file, options=[.m,.off,.obj]

    Returns
    -------
    nodes : list
        List of nodes. Each node is defined 2 or three coordinates
    faces : list
        List of faces, each face is defined via three indices
    """
    faces = []
    nodes = []
    with open(path) as f:
        for line in f:
            line = line.split()
            if file_type == ".m":
                if len(line) != 0 and line[0] == "Vertex":
                    out = [float(i) for i in line[2:]]
                    nodes.append(out)
                elif len(line) != 0 and line[0] == "Face":
                    out = [int(i) for i in line[2:]]
                    faces.append(out)
            elif file_type == ".obj":
                if len(line) != 0 and line[0] == "v":
                    out = [float(i) for i in line[1:]]
                    nodes.append(out)
                elif len(line) != 0 and line[0] == "f":
                    out = [int(i) for i in line[1:]]
                    faces.append(out)
            elif file_type == ".off":
                if len(line) == 3:
                    if line[0].find(".") != -1:
                        out = [float(i) for i in line[:]]
                        nodes.append(out)
                elif len(line) != 0 and line[0] == "3":
                    out = [int(i) for i in line[1:]]
                    faces.append(out)
            else:
                Exception("file_type must be from [.m,.obj,.off] ")

    return nodes, faces


def mesh_2_simplicial_complex(top_faces):
    """Transform a mesh to a simplicial complex.

    Parameters
    ----------
    top_faces : list
        List of faces.

    Returns
    -------
    _ : SimplicialComplex
        The simplicial complex of the input mesh.
    """
    return SimplicialComplex(top_faces)


def mesh_2_operators(faces, signed=False, norm_method="kipf", output_type="coo"):
    """Get the Laplacian and boundary operators of a mesh.

    Parameters
    ----------
    faces : a list of tuples
        Determines the mesh topology.
        Topologically, a trianglular mesh is completely determined by its faces.
    device : torch.device
        The torch.device contains a device type ('cpu' or 'cuda')

    Returns
    -------
    L0N : torch.tensor, shape=[N0,N0]
        The zero Laplacian matrix of the input mesh. This is also denoted by L0.
        N0 is the number of nodes in the input mesh.
    L1N : torch.tensor, shape=[N1,N1]
        The 1-Hodge Laplacian matrix of the input mesh. This is also denoted by L1.
        N1 is the number of edges in the input mesh.
    L2N : torch.tensor, shape [N2,N2]
        The 2-Hodge Laplacian matrix of the input mesh. This is also denoted by L2.
        N2 is the number of faces in the input mesh.
    B1N : torch.tensor, shape=[N0,N1]
        The boundary map partial_1 C^1 -> C^0.
        N0/N1 is the number of nodes/edges in the input mesh.
    B2N : torch.tensor, shape=[N1,N2]
        The boundary map partial_2 C^2 -> C^1.
        N1/N2 are the number of edges/faces in the input mesh.
    """
    hl = SimplicialComplex(faces)

    if norm_method in ["kipf", "xu", "row"]:
        print("computing the boundary and coboundary matrices..\n")
        B2 = hl.get_normalized_boundary_operator(
            d=2, signed=signed, normalization=norm_method
        )
        B1 = hl.get_normalized_boundary_operator(
            d=1, signed=signed, normalization=norm_method
        )
        B1T = hl.get_normalized_boundary_operator(
            d=1, signed=signed, normalization=norm_method
        )
        B2T = hl.get_normalized_boundary_operator(
            d=2, signed=signed, normalization=norm_method
        )

        print("computing the Hodge Laplacians matrices..\n")
        L0 = hl.get_normalized_hodge_laplacian(d=0, signed=signed)
        L1 = hl.get_normalized_hodge_laplacian(d=1, signed=signed)
        L2 = hl.get_normalized_hodge_laplacian(d=2, signed=signed)
        Adj0 = hl.get_normalized_higher_order_adj(d=0)
        Adj1 = hl.get_normalized_higher_order_adj(d=1)
        Coadj1 = hl.get_normalized_higher_order_coadj(d=1)
        Coadj2 = hl.get_normalized_higher_order_coadj(d=2)
        out = [Adj0, Adj1, Coadj1, Coadj2, L0, L1, L2, B1, B2, B1T, B2T]
        if output_type == "coo":
            return out
        elif output_type == "numpy" or output_type == "np":
            return [i.toarray() for i in out]
        elif output_type == "torch":
            return [coo_2_torch_tensor(i) for i in out]
        else:
            raise Exception("output_type must be from [numpy, torch, coo]")

    elif norm_method is None:
        print("computing the boundary matrices..\n")
        B2 = hl.get_boundary_operator(d=2, signed=signed)
        B1 = hl.get_boundary_operator(d=1, signed=signed)
        print("computing the Hodge Laplacians matrices..\n")
        L0 = hl.get_hodge_laplacian(0, signed)
        L1 = hl.get_hodge_laplacian(1, signed)
        L2 = hl.get_hodge_laplacian(2, signed)
        Adj0 = hl.get_higher_order_adj(d=0, signed=signed)
        Adj1 = hl.get_higher_order_adj(d=1, signed=signed)
        Coadj1 = hl.get_higher_order_coadj(d=1, signed=signed)
        Coadj2 = hl.get_higher_order_coadj(d=2, signed=signed)
        out = [Adj0, Adj1, Coadj1, Coadj2, L0, L1, L2, B1, B2]
        if output_type == "coo":
            return out + [None, None]  # output length consistent
        elif output_type == "numpy" or output_type == "np":
            return [i.toarray() for i in out] + [None, None]
        elif output_type == "torch":
            return [coo_2_torch_tensor(i) for i in out] + [None, None]
        else:
            raise Exception("output_type must be from [numpy, torch, coo]")
    else:
        raise Exception("norm_method must be from [kipf, xu, row, None]")


def _string_2_numbers(stringlist, data_type=np.float64):
    return [data_type(item) for item in stringlist]


def _read_txt_table(path, data_type=float):
    table = []
    with open(path) as f:
        for line in f:
            numbers_float = [str(x) for x in line.split()]
            pointlst = _string_2_numbers(numbers_float)
            table.append(pointlst)
    return table
