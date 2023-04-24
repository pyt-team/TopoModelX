import numpy as np
import scipy.linalg
import torch
from torch_geometric.data.data import Data
from torch_geometric.loader import DataLoader
from torch_sparse import SparseTensor
from toponetx import SimplicialComplex


def get_simplical_data_to_batch(simplices):
    # other available modes : gudhi--typically much faster
    Sc = SimplicialComplex(simplices, mode="gudhi")  

    B1 = Sc.incidence_matrix(1)  # B1: E(X)->V(X)
    B2 = Sc.incidence_matrix(2)  # B2: F(X)->E(X)

    L0 = Sc.hodge_laplacian_matrix(0)  # L0: V(X)->V(X), L0=D-A
    L1 = Sc.hodge_laplacian_matrix(1)  # L1: E(X)->E(X)
    L2 = Sc.hodge_laplacian_matrix(2)  # L2: F(X)->F(X)

    N0 = len(Sc.skeleton(rank=0))  # number of nodes
    N1 = len(Sc.skeleton(rank=1))  # number of edges
    N2 = len(Sc.skeleton(rank=2))  # number of faces

    x_v = torch.rand(N0, 3)
    x_e = torch.rand(N1, 3)
    x_f = torch.rand(N2, 3)
    return [B1, B2], [L0, L1, L2], [x_v, x_e, x_f]


def test_batch():
    simplices_A = [(0, 1, 2), (0, 4), (5,)]
    simplices_B = [(0, 4, 5), (1, 2), (3,)]

    Bs_A, Ls_A, xs_A = get_simplical_data_to_batch(simplices_A)
    Bs_B, Ls_B, xs_B = get_simplical_data_to_batch(simplices_B)

    B1_correct_A = [
        [-1, -1, -1, 0],
        [1, 0, 0, -1],
        [0, 1, 0, 1],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
    ]

    B1_correct_B = [
        [-1, -1, 0, 0],
        [0, 0, -1, 0],
        [0, 0, 1, 0],
        [0, 0, 0, 0],
        [1, 0, 0, -1],
        [0, 1, 0, 1],
    ]

    assert Bs_A[0].astype("int32").todense().tolist() == B1_correct_A
    assert Bs_B[0].astype("int32").todense().tolist() == B1_correct_B

    Bs_A = [SparseTensor.from_scipy(B) for B in Bs_A]
    Bs_B = [SparseTensor.from_scipy(B) for B in Bs_B]

    Ls_A = [SparseTensor.from_scipy(L) for L in Ls_A]
    Ls_B = [SparseTensor.from_scipy(L) for L in Ls_B]

    data_A = Data(
        xs=xs_A,
        c={  # c as abbreviation for complex; this dict could also be omitted
            "Bs": Bs_A,
            "Ls": Ls_A,
        },
        extra_attrs="Horse",
    )

    data_B = Data(
        xs=xs_B,
        c={"Bs": Bs_B, "Ls": Ls_B},
        extra_attrs="Sheep",
    )

    dataset = [data_A, data_B]
    loader = DataLoader(
        dataset, batch_size=2
    )  # creates _batch for named attr 'xs' in Data by default

    batch = None
    for batch in loader:
        break

    correct_batched_B1 = np.vstack(
        [np.array(B1_correct_A), np.array(B1_correct_B)]
    )
    assert np.allclose(correct_batched_B1, batch.c["Bs"][0].to_scipy("csr").todense())

    # batch vectors as usual in pyg, indicate which n, e or f is part of what complex
    # assert batch.xs[0].tolist() == 5 * [0] + 6 * [1]
    # assert batch.xs[1].tolist() == 4 * [0] + 4 * [1]
    # assert batch.xs[2].tolist() == 1 * [0] + 1 * [1]
