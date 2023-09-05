"""Unit tests for CWN."""
import itertools
import random

import torch

from topomodelx.nn.cell.cwn import CWN
from toponetx.classes.cell_complex import CellComplex

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCWN:
    """Unit tests for the CWN model class."""

    def test_forward(self):
        """Test the forward method of CWN."""
        faces = 14
        node_creation = 17
        nodes_per_face = 3
        node_features = 7
        edge_features = 10
        face_features = 6
        seed_value = 42
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        # Create a random cell complex of cells with length 3
        all_combinations = list(
            itertools.combinations(
                [x for x in range(1, node_creation + 1)], nodes_per_face
            )
        )
        # Shuffle the list of combinations
        random.shuffle(all_combinations)
        selected_combinations = all_combinations[:faces]
        cell_complex = CellComplex()
        cell_complex.add_cells_from(selected_combinations, rank=2)
        incidence_2 = cell_complex.incidence_matrix(rank=2)
        adjacency_1 = cell_complex.adjacency_matrix(rank=1)
        incidence_1_t = cell_complex.incidence_matrix(rank=1).T
        incidence_2 = (
            torch.from_numpy(incidence_2.todense()).float().to_sparse().to(device)
        )
        adjacency_1 = (
            torch.from_numpy(adjacency_1.todense()).float().to_sparse().to(device)
        )
        incidence_1_t = (
            torch.from_numpy(incidence_1_t.todense()).float().to_sparse().to(device)
        )
        node_feature_matrix = (
            torch.randn(len(cell_complex.nodes), node_features).float().to(device)
        )
        edge_feature_matrix = (
            torch.randn(len(cell_complex.edges), edge_features).float().to(device)
        )
        face_feature_matrix = (
            torch.randn(len(cell_complex.cells), face_features).float().to(device)
        )
        model = CWN(node_features, edge_features, face_features, 16, 1, 2).to(device)
        with torch.no_grad():
            forward_pass = model(
                node_feature_matrix,
                edge_feature_matrix,
                face_feature_matrix,
                adjacency_1,
                incidence_2,
                incidence_1_t,
            )
        assert torch.any(
            torch.isclose(
                input=forward_pass, other=torch.tensor([-0.7850]).to(device), rtol=1e-02
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of CWN."""
        node_features = 7
        edge_features = 30
        face_features = 10
        model = CWN(node_features, edge_features, face_features, 16, 1, 2).to(device)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
