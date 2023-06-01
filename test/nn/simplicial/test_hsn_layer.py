"""Test the HSN layer."""

import torch

from topomodelx.nn.simplicial.hsn_layer import HSNLayer


class TestHSNLayer:
    """Test the HSN layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        channels = 5
        n_nodes = 10
        n_edges = 20
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        adjacency_0 = torch.randint(0, 2, (n_nodes, n_nodes)).float()

        x_0 = torch.randn(n_nodes, channels)

        hsn = HSNLayer(channels)
        output = hsn.forward(x_0, incidence_1, adjacency_0)

        assert output.shape == (n_nodes, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        hsn = HSNLayer(channels)
        hsn.reset_parameters()

        for module in hsn.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
