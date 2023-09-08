"""Unit tests for SCA Model."""
import itertools
import random

import torch
from toponetx.classes import SimplicialComplex

from topomodelx.nn.simplicial.sca_cmps import SCACMPS


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
        simplicial_complex = SimplicialComplex()
        for simplex in selected_combinations:
            simplicial_complex.add_simplex(simplex)

        x_0 = torch.randn(15, 2)
        x_1 = torch.randn(35, 2)
        x_2 = torch.randn(14, 2)
        down_lap1 = simplicial_complex.down_laplacian_matrix(rank=1)
        down_lap2 = simplicial_complex.down_laplacian_matrix(rank=2)
        incidence_1t = simplicial_complex.incidence_matrix(rank=1).T
        incidence_2t = simplicial_complex.incidence_matrix(rank=2).T
        down_lap1 = torch.from_numpy(down_lap1.todense()).to_sparse()
        down_lap2 = torch.from_numpy(down_lap2.todense()).to_sparse()
        incidence_1t = torch.from_numpy(incidence_1t.todense()).to_sparse()
        incidence_2t = torch.from_numpy(incidence_2t.todense()).to_sparse()
        channels_list = [x_0.shape[-1], x_1.shape[-1], x_2.shape[-1]]
        complex_dim = 3
        model = SCACMPS(
            channels_list=channels_list,
            complex_dim=complex_dim,
            n_classes=1,
            n_layers=3,
            att=False,
        )
        x_list = [x_0, x_1, x_2]
        down_lap_list = [down_lap1, down_lap2]
        incidence_t_list = [incidence_1t, incidence_2t]
        forward_pass = model(x_list, down_lap_list, incidence_t_list)
        assert torch.any(
            torch.isclose(forward_pass, torch.tensor([-4.8042]), rtol=1e-02)
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCA."""
        model = SCACMPS(
            [2, 2, 2],
            2,
            n_classes=1,
            n_layers=3,
            att=False,
        )
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
