"""Unit tests for SCCNN Model."""

import itertools
import random

import toponetx as tnx
import torch

from topomodelx.nn.simplicial.scconv import SCConv
from topomodelx.utils.sparse import from_sparse


class TestSCConv:
    """Unit tests for the SCConv model class."""

    def test_forward(self):
        """Test the forward method of SCConv."""
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
        # Some nodes might not be selected at all in the combinations above
        x_0 = torch.randn(simplicial_complex.shape[0], 2)
        x_1 = torch.randn(simplicial_complex.shape[1], 2)
        x_2 = torch.randn(faces, 2)

        incidence_1_norm = from_sparse(simplicial_complex.incidence_matrix(1))
        incidence_1 = from_sparse(simplicial_complex.coincidence_matrix(1))
        incidence_2_norm = from_sparse(simplicial_complex.incidence_matrix(2))
        incidence_2 = from_sparse(simplicial_complex.coincidence_matrix(2))
        adjacency_up_0_norm = from_sparse(simplicial_complex.up_laplacian_matrix(0))
        adjacency_up_1_norm = from_sparse(simplicial_complex.up_laplacian_matrix(1))
        adjacency_down_1_norm = from_sparse(simplicial_complex.down_laplacian_matrix(1))
        adjacency_down_2_norm = from_sparse(simplicial_complex.down_laplacian_matrix(2))

        in_channels = x_0.shape[1]
        n_layers = 2
        model = SCConv(
            node_channels=in_channels,
            n_layers=n_layers,
        )

        with torch.no_grad():
            forward_pass = model(
                x_0,
                x_1,
                x_2,
                incidence_1,
                incidence_1_norm,
                incidence_2,
                incidence_2_norm,
                adjacency_up_0_norm,
                adjacency_up_1_norm,
                adjacency_down_1_norm,
                adjacency_down_2_norm,
            )
        assert torch.any(
            torch.isclose(
                forward_pass[0][0],
                torch.tensor(
                    [
                        0.8847,
                        0.9963,
                    ]
                ),
                rtol=1e-02,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCConv."""
        model = SCConv(4, 2)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
