"""Unit tests for SCCN Model."""

import itertools
import random

import toponetx as tnx
import torch

from topomodelx.nn.simplicial.sccn import SCCN
from topomodelx.utils.sparse import from_sparse


class TestSCCN:
    """Unit tests for the SCCN model class."""

    def test_forward(self):
        """Test the forward method of SCCN."""
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
        x_2 = torch.randn(simplicial_complex.shape[2], 2)
        features = {"rank_0": x_0, "rank_1": x_1, "rank_2": x_2}
        max_rank = 2

        def sparse_to_torch(X):
            return from_sparse(X)

        incidences = {
            f"rank_{r}": sparse_to_torch(simplicial_complex.incidence_matrix(rank=r))
            for r in range(1, max_rank + 1)
        }
        adjacencies = {}
        adjacencies["rank_0"] = (
            sparse_to_torch(simplicial_complex.adjacency_matrix(rank=0))
            + torch.eye(simplicial_complex.shape[0]).to_sparse()
        )
        for r in range(1, max_rank):
            adjacencies[f"rank_{r}"] = (
                sparse_to_torch(
                    simplicial_complex.adjacency_matrix(rank=r)
                    + simplicial_complex.coadjacency_matrix(rank=r)
                )
                + 2 * torch.eye(simplicial_complex.shape[r]).to_sparse()
            )
        adjacencies[f"rank_{max_rank}"] = (
            sparse_to_torch(simplicial_complex.coadjacency_matrix(rank=max_rank))
            + torch.eye(simplicial_complex.shape[max_rank]).to_sparse()
        )
        channels_nodes = x_0.shape[-1]
        model = SCCN(
            channels=channels_nodes,
            max_rank=max_rank,
            n_layers=5,
            update_func="sigmoid",
        )
        forward_pass = model(features, incidences, adjacencies)
        assert torch.any(
            torch.isclose(
                forward_pass["rank_0"],
                torch.tensor(
                    [
                        [0.9937, 0.0502],
                        [0.9997, 0.0355],
                        [0.9991, 0.0189],
                        [1.0000, 0.0550],
                        [0.9991, 0.2027],
                        [0.9996, 0.1193],
                        [0.9859, 0.2987],
                        [0.9984, 0.3094],
                        [0.9977, 0.1353],
                        [1.0000, 0.1991],
                        [0.9998, 0.3353],
                        [1.0000, 0.2689],
                        [0.9999, 0.4945],
                        [0.9994, 0.4820],
                        [0.9999, 0.4628],
                    ]
                ),
                rtol=1e-02,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of SCCN."""
        model = SCCN(7, 3, 3)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
