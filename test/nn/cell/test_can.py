"""Unit tests for CAN"""

from toponetx.classes.cell_complex import CellComplex
from torch_geometric.utils.convert import to_networkx
import torch
import networkx as nx
import torch.nn.functional as F
from topomodelx.nn.cell.can_layer import CANLayer, PoolLayer, LiftLayer
from topomodelx.nn.cell.can import CAN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class TestCAN:
    """Unit tests for the CAN model class."""

    def test_forward(self):
        """Test the forward method of CAN."""

        nodes = 17
        edges = 38
        node_features = 7
        edge_features = 4
        torch.manual_seed(42)

        model = CAN(
            node_features,
            edge_features,
            32,
            dropout=0.5,
            heads=2,
            num_classes=2,
            n_layers=2,
            att_lift=True,
        ).to(device)
        adjacency = nx.gnm_random_graph(nodes, edges / 2, seed=42)
        # Convert the graph to an adjacency matrix
        adjacency = torch.tensor(nx.to_numpy_matrix(adjacency))
        feature_nodes = torch.randn(nodes, node_features).to(device)
        feature_edges = torch.randn(edges, edge_features).to(device)
        lower_neighborhood = torch.randn(edges, edges).to(device)
        upper_neighborhood = torch.randn(edges, edges).to(device)
        lower_neighborhood = lower_neighborhood.to_sparse().float().to(device)
        upper_neighborhood = upper_neighborhood.to_sparse().float().to(device)
        adjacency = adjacency.to_sparse().float().to(device)
        with torch.no_grad():
            forward_pass = model(
                feature_nodes,
                feature_edges,
                adjacency.float().to_sparse().to(device),
                lower_neighborhood,
                upper_neighborhood,
            )
        assert torch.any(
            torch.isclose(
                input=forward_pass,
                other=torch.tensor([-0.0852, -0.3328]).to(device),
                rtol=1e-02,
            )
        )

    def test_forward_without_attn(self):
        """Test the forward method of CAN.(without attn)"""
        nodes = 17
        edges = 38
        node_features = 7
        edge_features = 4

        torch.manual_seed(42)
        model = CAN(
            node_features,
            edge_features,
            32,
            dropout=0.5,
            heads=2,
            num_classes=2,
            n_layers=2,
            att_lift=False,
        ).to(device)
        adjacency = nx.gnm_random_graph(nodes, edges / 2, seed=42)
        # Convert the graph to an adjacency matrix
        adjacency = torch.tensor(nx.to_numpy_matrix(adjacency))
        feature_nodes = torch.randn(nodes, node_features).to(device)
        feature_edges = torch.randn(edges, edge_features).to(device)
        lower_neighborhood = torch.randn(edges, edges).to(device)
        upper_neighborhood = torch.randn(edges, edges).to(device)
        lower_neighborhood = lower_neighborhood.to_sparse().float().to(device)
        upper_neighborhood = upper_neighborhood.to_sparse().float().to(device)
        adjacency = adjacency.to_sparse().float().to(device)
        with torch.no_grad():
            forward_pass = model(
                feature_nodes,
                feature_edges,
                adjacency.float().to_sparse().to(device),
                lower_neighborhood,
                upper_neighborhood,
            )
        assert torch.any(
            torch.isclose(
                input=forward_pass,
                other=torch.tensor([-0.0696, 0.0559]).to(device),
                rtol=1e-02,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of CAN."""
        node_features = 7
        edge_features = 4
        model = CAN(
            node_features,
            edge_features,
            32,
            dropout=0.5,
            heads=2,
            num_classes=2,
            n_layers=2,
            att_lift=False,
        ).to(device)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
