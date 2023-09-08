"""Unit tests for Dist2Cycke Model."""
import numpy as np
import torch
from toponetx.classes import SimplicialComplex

from topomodelx.nn.simplicial.dist2cycle import Dist2Cycle


class TestDist2Cycle:
    """Unit tests for the Dist2Cycle model class."""

    def test_forward(self):
        """Test the forward method of Dist2Cycle."""
        edge_set = [[1, 2], [1, 3], [2, 5], [3, 5]]
        face_set = [[2, 3, 4], [2, 4, 5]]

        torch.manual_seed(42)
        simplicial_complex = SimplicialComplex(edge_set + face_set)
        laplacian_down_1 = simplicial_complex.down_laplacian_matrix(rank=1).todense()
        adjacency_1 = simplicial_complex.adjacency_matrix(rank=1).todense()
        laplacian_down_1_inv = np.linalg.pinv(
            laplacian_down_1 + np.eye(laplacian_down_1.shape[0])
        )  # test inverse
        adjacency_1 = torch.from_numpy(adjacency_1).float().to_sparse()
        laplacian_inv = torch.from_numpy(laplacian_down_1_inv).float().to_sparse()
        res = adjacency_1 * laplacian_inv
        x_1e = res.to_sparse()
        model = Dist2Cycle(8, 2)
        assert torch.any(
            torch.isclose(
                model(x_1e, laplacian_inv, adjacency_1)[0],
                torch.tensor([0.4174, 0.5826]),
                rtol=1e-2,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of Dist2Cycle."""
        model = Dist2Cycle(8, 2)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
