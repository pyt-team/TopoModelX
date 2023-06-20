"""Test the SCoNe layer."""

import torch

from topomodelx.nn.simplicial.scone_layer import SCoNeLayer


class TestSCoNeLayer:
    """Test the SCoNe layer."""

    def test_forward(self):
        """Test the forward pass of the SCoNe layer."""
        in_channels = 8
        out_channels = 16
        n_nodes = 10
        n_edges = 20
        n_triangles = 30 

        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        incidence_2 = torch.randint(0, 2, (n_edges, n_triangles)).float()

        x_0 = torch.randn(n_edges, in_channels)

        scone = SCoNeLayer(in_channels, out_channels)
        output = scone.forward(x_0, incidence_1, incidence_2)

        assert output.shape == (n_edges, out_channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        in_channels = 8
        out_channels = 16 

        scone = SCoNeLayer(in_channels, out_channels)
        scone.reset_parameters()

        assert scone.weight_0.shape == (in_channels, out_channels)
        assert scone.weight_1.shape == (in_channels, out_channels)
        assert scone.weight_2.shape == (in_channels, out_channels)

        scone.reset_parameters(gain=0)

        torch.testing.assert_close(scone.weight_0, torch.zeros_like(scone.weight_0))
        torch.testing.assert_close(scone.weight_1, torch.zeros_like(scone.weight_1))
        torch.testing.assert_close(scone.weight_2, torch.zeros_like(scone.weight_2))
