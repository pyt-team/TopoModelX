"""Unit tests for CCXN"""

from toponetx.classes.cell_complex import CellComplex
import torch
import networkx as nx
import torch.nn.functional as F
from topomodelx.nn.cell.ccxn_layer import CCXNLayer
from topomodelx.nn.cell.ccxn import CCXN
import random
import itertools

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
class TestCCXN:
    """Unit tests for the CCXN model class."""

    def test_forward(self):
        """Test the forward method of CCXN."""

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
        all_combinations = list(itertools.combinations([x for x in range(1,node_creation+1)], nodes_per_face))
        # Shuffle the list of combinations
        random.shuffle(all_combinations)
        selected_combinations = all_combinations[:faces]
        cell_complex = CellComplex()
        cell_complex.add_cells_from(selected_combinations,rank = 2)
        incidence_2 = cell_complex.incidence_matrix(rank=2)
        adjacency_0 = cell_complex.adjacency_matrix(rank=0)
        incidence_2_t = cell_complex.incidence_matrix(rank=2).T
        incidence_2 = torch.from_numpy(incidence_2.todense()).float().to_sparse().to(device)
        adjacency_0 = torch.from_numpy(adjacency_0.todense()).float().to_sparse().to(device)
        incidence_2_t = torch.from_numpy(incidence_2_t.todense()).float().to_sparse().to(device)
        node_feature_matrix = torch.randn(len(cell_complex.nodes),node_features).float().to(device)
        edge_feature_matrix = torch.randn(len(cell_complex.edges),edge_features).float().to(device)
        model = CCXN(node_features,edge_features,face_features,1,True).to(device)
        with torch.no_grad():
            forward_pass = model(node_feature_matrix,edge_feature_matrix,adjacency_0, incidence_2_t)
        assert torch.any(torch.isclose(input = forward_pass,other = torch.tensor([-1.7816]).to(device),rtol=1e-02))

    def test_reset_parameters(self):
        """Test the reset_parameters method of CCXN."""
        node_features = 7
        edge_features = 30
        face_features = 10
        model = CCXN(node_features,edge_features,face_features,1,True).to(device)
        for layer in model.children():
            if hasattr(layer, 'reset_parameters'):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()