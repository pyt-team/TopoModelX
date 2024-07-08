"""Unit tests for SCN2 Model."""

import itertools
import random

import toponetx as tnx
import torch

from topomodelx.nn.simplicial.scn2 import SCN2
from topomodelx.utils.sparse import from_sparse


class TestSCN2:
    """Unit tests for the SCN2 model class."""

    def test_forward(self):
        """Test the forward method of SCN2."""
        faces = 14
        node_creation = 17
        nodes_per_face = 3
        seed_value = 42
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        # Create a random cell complex of cells with length 3
        all_combinations = list(
            itertools.combinations(
                [x for x in range(1, node_creation + 1)], nodes_per_face
            )
        )
        random.shuffle(all_combinations)
        selected_combinations = all_combinations[:faces]
        simplicial_complex = tnx.SimplicialComplex()
        for simplex in selected_combinations:
            simplicial_complex.add_simplex(simplex)

        x_0 = torch.randn(simplicial_complex.shape[0], 2)
        x_1 = torch.randn(simplicial_complex.shape[1], 2)
        x_2 = torch.randn(faces, 2)
        laplacian_0 = simplicial_complex.normalized_laplacian_matrix(rank=0)
        laplacian_1 = simplicial_complex.normalized_laplacian_matrix(rank=1)
        laplacian_2 = simplicial_complex.normalized_laplacian_matrix(rank=2)

        laplacian_0 = from_sparse(laplacian_0)
        laplacian_1 = from_sparse(laplacian_1)
        laplacian_2 = from_sparse(laplacian_2)
        in_channels_0 = x_0.shape[1]
        in_channels_1 = x_1.shape[1]
        in_channels_2 = x_2.shape[1]
        model = SCN2(in_channels_0, in_channels_1, in_channels_2)
        forward_pass = model(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)
        with torch.no_grad():
            forward_pass = model(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)
        assert torch.any(
            torch.isclose(
                forward_pass[0][0], torch.tensor([0.1980, 0.0922]), rtol=1e-02
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCN2."""
        model = SCN2(2, 2, 2, 1)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
