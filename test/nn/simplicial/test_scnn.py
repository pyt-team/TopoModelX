"""Unit tests for SCNN Model."""
import itertools
import random

import numpy as np
import torch
from toponetx.classes import SimplicialComplex

from topomodelx.nn.simplicial.scnn import SCNN
from topomodelx.utils.sparse import from_sparse


class TestSCNN:
    """Unit tests for the SCNN model class."""

    def test_forward(self):
        """Test the forward method of SCNN."""
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
        x_1 = torch.randn(simplicial_complex.shape[1], 2)
        in_channels_1 = x_1.shape[1]

        incidence_1 = simplicial_complex.incidence_matrix(rank=1)
        incidence_2 = simplicial_complex.incidence_matrix(rank=2)
        laplacian_0 = simplicial_complex.hodge_laplacian_matrix(rank=0, weight=True)
        laplacian_down_1 = simplicial_complex.down_laplacian_matrix(rank=1, weight=True)
        laplacian_up_1 = simplicial_complex.up_laplacian_matrix(rank=1)
        laplacian_2 = simplicial_complex.hodge_laplacian_matrix(rank=2, weight=True)

        incidence_1 = from_sparse(incidence_1)
        incidence_2 = from_sparse(incidence_2)
        laplacian_0 = from_sparse(laplacian_0)
        laplacian_down_1 = from_sparse(laplacian_down_1)
        laplacian_up_1 = from_sparse(laplacian_up_1)
        laplacian_2 = from_sparse(laplacian_2)
        conv_order_down = 2
        conv_order_up = 2
        intermediate_channels = 4
        out_channels = intermediate_channels
        num_layers = 2
        in_channels = in_channels_1

        def get_simplicial_features(dataset, rank):
            if rank == 0:
                which_feat = "node_feat"
            elif rank == 1:
                which_feat = "edge_feat"
            elif rank == 2:
                which_feat = "face_feat"
            else:
                raise ValueError(
                    "input dimension must be 0, 1 or 2, because features are supported on nodes, edges and faces"
                )

            x = []
            for _, v in dataset.get_simplex_attributes(which_feat).items():
                x.append(v)

            x = torch.tensor(np.stack(x))
            return x

        model = SCNN(
            in_channels=in_channels,
            intermediate_channels=intermediate_channels,
            out_channels=out_channels,
            conv_order_down=conv_order_down,
            conv_order_up=conv_order_up,
            n_layers=num_layers,
            aggr=True,
        )
        with torch.no_grad():
            forward_pass = model(x_1, laplacian_down_1, laplacian_up_1)
        assert torch.any(
            torch.isclose(forward_pass, torch.tensor([137.2366]), rtol=1e-02)
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCNN."""
        model = SCNN(2, 2, 2, 2, 2, 1)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
