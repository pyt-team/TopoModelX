"""Unit tests for HSN Model."""
import numpy as np
import numpy.linalg as npla
import torch
from toponetx.classes import SimplicialComplex as sc

from topomodelx.nn.simplicial.hsn import HSN


class TestDist2Layer:
    """Unit tests for the HSN model class."""

    def test_forward(self):
        """Test the forward method of HSN."""
        edge_set = [[1, 2], [1, 3], [2, 5], [3, 5]]
        face_set = [[2, 3, 4], [2, 4, 5]]

        torch.manual_seed(42)
        ex2_sc = sc(edge_set + face_set)
        ld = ex2_sc.down_laplacian_matrix(rank=1).todense()
        A = ex2_sc.adjacency_matrix(rank=1).todense()
        L_tilde_pinv = npla.pinv(ld + np.eye(ld.shape[0]))  # test inverse
        adjacency = torch.from_numpy(A).float().to_sparse()
        Linv = torch.from_numpy(L_tilde_pinv).float().to_sparse()
        res = adjacency * Linv
        x_1e = res.to_sparse()
        model = HSN(8, 2)
        assert torch.any(
            torch.isclose(
                model(x_1e, Linv, adjacency)[0],
                torch.tensor([0.4077, 0.5923]),
                rtol=1e-2,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of HSN."""
        model = HSN(8, 2)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
