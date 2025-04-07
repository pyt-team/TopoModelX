"""Test the SNN layer."""

import torch

from topomodelx.nn.simplicial.snn_layer import SNNLayer


class TestSNNLayer:
    """Test the SNN layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        in_channels = 5
        out_channels = 5
        n_nodes = 10
        K = 5
        lapl_0 = torch.randint(0, 2, (n_nodes, n_nodes)).float()

        x_0 = torch.randn(n_nodes, in_channels)

        snn = SNNLayer(in_channels, out_channels, K)
        output = snn.forward(x_0, lapl_0)

        assert output.shape == (n_nodes, out_channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        in_channels = 5
        out_channels = 5
        K = 5

        snn = SNNLayer(in_channels, out_channels, K)
        snn.reset_parameters()

        for module in snn.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
