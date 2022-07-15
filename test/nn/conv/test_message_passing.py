import torch
from torch_geometric.nn import MessagePassing
from torch_sparse import SparseTensor

from torch_topo.topology import SimplicialComplex


def get_simplical_data_to_batch(simplices):
    HL = SimplicialComplex(
        simplices, mode="gudhi"
    )  # other available modes : gudhi--typically much faster

    B1 = HL.get_boundary_operator(1)  # B1: E(X)->V(X)
    B2 = HL.get_boundary_operator(2)  # B2: F(X)->E(X)

    L0 = HL.get_hodge_laplacian(0)  # L0: V(X)->V(X), L0=D-A
    L1 = HL.get_hodge_laplacian(1)  # L1: E(X)->E(X)
    L2 = HL.get_hodge_laplacian(2)  # L2: F(X)->F(X)

    N0 = len(HL.n_faces(0))  # number of nodes
    N1 = len(HL.n_faces(1))  # number of edges
    N2 = len(HL.n_faces(2))  # number of faces

    x_v = torch.rand(N0, 3)
    x_e = torch.rand(N1, 3)
    x_f = torch.rand(N2, 3)
    return [B1, B2], [L0, L1, L2], [x_v, x_e, x_f]


class DummyMessagePassing(MessagePassing):
    """
    Simple DummyMessagePassing that simulates matrix multiplication B @ x
    """

    def forward(self, B, x):
        return self.propagate(B, x=x, values=B.storage.value())

    def message(self, x_j, values):
        return values.view(-1, 1) * x_j


def test_dummy_mp():
    simplices = [(0, 1, 2), (0, 4), (5,)]

    Bs, Ls, xs = get_simplical_data_to_batch(simplices)
    B1_scipy = Bs[0]

    Bs = [SparseTensor.from_scipy(B) for B in Bs]
    B1_tensor = Bs[0]

    dummy_mp = DummyMessagePassing()
    assert torch.allclose(
        torch.from_numpy(B1_scipy.todense()) @ xs[1], dummy_mp(B1_tensor, xs[1])
    )
