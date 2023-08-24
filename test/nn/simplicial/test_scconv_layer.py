"""Test the SCConv layer."""
import torch

from topomodelx.nn.simplicial.scconv_layer import SCConvLayer


class TestSCConvLayer:
    """Test the SCConv layer."""

    def test_forward(self):
        """Test the forward pass of the SCConv layer."""
        node_channels = 6
        edge_channels = 9
        face_channels = 15

        n_nodes = 10
        n_edges = 20
        n_faces = 30

        x_0 = torch.randn(n_nodes, node_channels)
        x_1 = torch.randn(n_edges, edge_channels)
        x_2 = torch.randn(n_faces, face_channels)
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        # incidence_1_norm = torch.randint(0, 2, (n_nodes, n_edges)).float()
        incidence_1_norm = torch.randint(0, 2, (n_edges, n_nodes)).float()

        incidence_2 = torch.randint(0, 2, (n_edges, n_faces)).float()
        # incidence_2_norm = torch.randint(0, 2, (n_edges, n_faces)).float()
        incidence_2_norm = torch.randint(0, 2, (n_faces, n_edges)).float()

        adjacency_up_0_norm = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        adjacency_up_1_norm = torch.randint(0, 2, (n_edges, n_edges)).float()
        adjacency_down_1_norm = torch.randint(0, 2, (n_edges, n_edges)).float()
        adjacency_down_2_norm = torch.randint(0, 2, (n_faces, n_faces)).float()

        scconv = SCConvLayer(node_channels, edge_channels, face_channels)
        out1, out2, out3 = scconv.forward(
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

        # to be updated
        assert out1.shape == (n_nodes, node_channels)
        assert out2.shape == (n_edges, edge_channels)
        assert out3.shape == (n_faces, face_channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        node_channels = 6
        edge_channels = 10
        face_channels = 7

        scconv = SCConvLayer(node_channels, edge_channels, face_channels)
        scconv.reset_parameters()

        for module in scconv.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
