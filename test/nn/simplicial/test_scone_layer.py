"""Test the SCoNe Layer."""

import torch

from topomodelx.nn.simplicial.scone_layer import SCoNeLayer


class TestSCoNeLayer:
    """Test the SCoNe Layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        channels = 5
        n_edges = 20
        up_lap1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        down_lap1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        iden = torch.eye(n_edges)

        x_1 = torch.randn(n_edges, channels)

        scone = SCoNeLayer(channels)
        output = scone.forward(x_1, up_lap1, down_lap1, iden)

        assert output.shape == (n_edges, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        scone = SCoNeLayer(channels)
        scone.reset_parameters()

        for module in scone.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
