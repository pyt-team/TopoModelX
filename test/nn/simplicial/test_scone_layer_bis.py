"""Unit tests for SconeBis Model."""
import itertools
import random

import torch

from topomodelx.nn.simplicial.scone_bis import SCoNeNN
from toponetx.classes import SimplicialComplex as sc


class TestSconeBis:
    """Unit tests for the SconeBis model class."""

    def test_forward(self):
        """Test the forward method of SconeBis."""
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
        simplicial_complex = sc()
        for simplex in selected_combinations:
            simplicial_complex.add_simplex(simplex)
        x_1 = torch.randn(35, 1)
        up_lap1 = simplicial_complex.up_laplacian_matrix(rank=1)
        down_lap1 = simplicial_complex.down_laplacian_matrix(rank=1)
        up_lap1 = torch.from_numpy(up_lap1.todense()).to_sparse()
        down_lap1 = torch.from_numpy(down_lap1.todense()).to_sparse()
        dim = 35
        iden = torch.eye(dim).to_sparse()
        edge_channels = 1
        model = SCoNeNN(
            channels=edge_channels,
            n_layers=10,
        )
        with torch.no_grad():
            forward_pass = model(x_1, up_lap1, down_lap1, iden)
        assert torch.any(
            torch.isclose(forward_pass[0], torch.tensor([0.5440, 0.4560]), rtol=1e-02)
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SconeBis."""
        model = SCoNeNN(
            channels=2,
            n_layers=10,
        )
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
