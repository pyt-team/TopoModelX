"""Unit tests for HSN Model."""

import numpy as np
import toponetx as tnx
import torch

from topomodelx.nn.simplicial.hsn import HSN


class TestHSN:
    """Unit tests for the HSN model class."""

    def test_forward(self):
        """Test the forward method of HSN."""
        edge_set = [[1, 2], [1, 3], [2, 5], [3, 5]]
        face_set = [[2, 3, 4], [2, 4, 5]]

        torch.manual_seed(42)
        simplicial_complex = tnx.SimplicialComplex(edge_set + face_set)
        laplacian_down_1 = simplicial_complex.down_laplacian_matrix(rank=1).todense()
        adjacency_1 = simplicial_complex.adjacency_matrix(rank=1).todense()
        laplacian_down_1_inv = np.linalg.pinv(
            laplacian_down_1 + np.eye(laplacian_down_1.shape[0])
        )  # test inverse
        adjacency_1 = torch.from_numpy(adjacency_1).float().to_sparse()
        laplacian_inv = torch.from_numpy(laplacian_down_1_inv).float().to_sparse()
        res = adjacency_1 * laplacian_inv
        x_1e = res.to_sparse()
        model = HSN(8, 2)
        assert torch.any(
            torch.isclose(
                model(x_1e, laplacian_inv, adjacency_1)[0],
                torch.tensor(
                    [0.6942, 0.5649, 0.3680, 0.3968, 0.4345, 0.4687, 0.5999, 0.4920]
                ),
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
