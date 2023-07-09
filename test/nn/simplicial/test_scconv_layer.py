"""Test the SCConv layer."""
import torch

from topomodelx.nn.simplicial.scconv_layer import SCConvLayer


class TestSCConvLayer:
    """Test the SCConv layer."""

    def test_forward(self):
        """Test the forward pass of the SCConv layer."""
        channels = 5
        n_nodes = 10
        n_edges = 20
        n_faces = 30

        x_0 = torch.randn(n_nodes, channels)
        x_1 = torch.randn(n_edges, channels)
        x_2 = torch.randn(n_faces, channels)
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        incidence_1_norm = torch.randint(0, 2, (n_nodes, n_edges)).float()
        incidence_2 = None
        incidence_2_norm = None
        adjacency_up_0_norm = None
        adjacency_up_1_norm = None
        adjacency_down_1_norm = None
        adjacency_down_2_norm = None

        scconv = SCConvLayer(channels)
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
        assert out1.shape == (n_nodes, channels)
        assert out2.shape == (n_edges, channels)
        assert out3.shape == (n_faces, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        scconv = SCConvLayer(channels)
        scconv.reset_parameters()

        for module in scconv.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
