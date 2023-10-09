"""Unit tests for SAN Model."""
import itertools
import random

import torch
from toponetx.classes import SimplicialComplex

from topomodelx.nn.simplicial.san import SAN
from topomodelx.utils.sparse import from_sparse


class TestSAN:
    """Unit tests for the SAN model class."""

    def test_forward(self):
        """Test the forward method of SAN."""
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
        simplicial_complex = SimplicialComplex()
        for simplex in selected_combinations:
            simplicial_complex.add_simplex(simplex)
        x_1 = torch.randn(35, 2)
        x_0 = torch.randn(15, 2)
        incidence_0_1 = from_sparse(simplicial_complex.incidence_matrix(1))
        x = x_1 + torch.sparse.mm(incidence_0_1.T, x_0)
        in_channels = x.shape[-1]
        hidden_channels = 16
        out_channels = 2
        model = SAN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            out_channels=out_channels,
            n_layers=1,
        )
        laplacian_down_1 = from_sparse(simplicial_complex.down_laplacian_matrix(rank=1))
        laplacian_up_1 = from_sparse(simplicial_complex.up_laplacian_matrix(rank=1))

        assert torch.any(
            torch.isclose(
                model(x, laplacian_up_1, laplacian_down_1)[0],
                torch.tensor([0.7727, 0.2389]),
                rtol=1e-02,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SAN."""
        model = SAN(7, 3, 3)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
