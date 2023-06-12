"""Test the template layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.dhgnn_layer import DHGNNLayer


class TestDHGNNLayer:
    """Test the DHGNN layer."""

    @pytest.fixture
    def dhgnn_layer(self):
        """Return a DHGNN layer."""
        in_channels = 10
        intermediate_channels = 20
        return DHGNNLayer(in_channels, intermediate_channels)

    def test_forward(self, dhgnn_layer):
        """Test the forward pass of the DHGNN layer."""
        x_2 = torch.randn(3, 10)
        incidence_2 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        output = dhgnn_layer.forward(x_2, incidence_2)
        assert output.shape == torch.Size([])
