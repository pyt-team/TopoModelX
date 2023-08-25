"""Unit tests for the SCN2Layer class."""

import torch

from topomodelx.nn.simplicial.scn2_layer import SCN2Layer


class TestSCN2Layer:
    """Unit tests for the SCN2Layer class."""

    def test_forward(self):
        """Test the forward method of SCN2Layer."""
        n_0_cells = 10
        n_1_cells = 20
        n_2_cells = 30
        channels_0 = 10
        channels_1 = 20
        channels_2 = 30

        x_0 = torch.randn(n_0_cells, channels_0)
        x_1 = torch.randn(n_1_cells, channels_1)
        x_2 = torch.randn(n_2_cells, channels_2)
        laplacian_0 = torch.randn(n_0_cells, n_0_cells)
        laplacian_1 = torch.randn(n_1_cells, n_1_cells)
        laplacian_2 = torch.randn(n_2_cells, n_2_cells)

        scn_layer = SCN2Layer(
            in_channels_0=channels_0,
            in_channels_1=channels_1,
            in_channels_2=channels_2,
        )
        x_0, x_1, x_2 = scn_layer.forward(
            x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2
        )

        assert x_0.shape == (n_0_cells, channels_0)
        assert x_1.shape == (n_1_cells, channels_1)
        assert x_2.shape == (n_2_cells, channels_2)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels_0 = 10
        channels_1 = 20
        channels_2 = 30

        scn = SCN2Layer(channels_0, channels_1, channels_2)
        scn.reset_parameters()

        for module in scn.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
