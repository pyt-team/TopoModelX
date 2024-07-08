"""Unit tests for SCA Model."""

import itertools
import random

import toponetx as tnx
import torch

from topomodelx.nn.simplicial.sca_cmps import SCACMPS
from topomodelx.utils.sparse import from_sparse


class TestSCA:
    """Unit tests for the SCA model class."""

    def test_forward(self):
        """Test the forward method of SCA."""
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

        x_0 = torch.randn(15, 2)
        x_1 = torch.randn(35, 2)
        x_2 = torch.randn(14, 2)
        down_lap1 = simplicial_complex.down_laplacian_matrix(rank=1)
        down_lap2 = simplicial_complex.down_laplacian_matrix(rank=2)
        incidence_1t = simplicial_complex.incidence_matrix(rank=1).T
        incidence_2t = simplicial_complex.incidence_matrix(rank=2).T
        down_lap1 = from_sparse(down_lap1)
        down_lap2 = from_sparse(down_lap2)
        incidence_1t = from_sparse(incidence_1t)
        incidence_2t = from_sparse(incidence_2t)
        channels_list = [x_0.shape[-1], x_1.shape[-1], x_2.shape[-1]]
        complex_dim = 3
        model = SCACMPS(
            in_channels_all=channels_list,
            complex_dim=complex_dim,
            n_layers=3,
            att=False,
        )
        x_list = [x_0, x_1, x_2]
        down_lap_list = [down_lap1, down_lap2]
        incidence_t_list = [incidence_1t, incidence_2t]
        forward_pass = model(x_list, down_lap_list, incidence_t_list)
        assert torch.any(
            torch.isclose(
                forward_pass[0][0], torch.tensor([1.9269, 1.4873]), rtol=1e-02
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCA."""
        model = SCACMPS(
            [2, 2, 2],
            2,
            n_layers=3,
            att=False,
        )
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
