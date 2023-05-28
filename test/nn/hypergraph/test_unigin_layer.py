"""Test the UniGIN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unigin_layer import UniGINLayer


class TestUniGINLayer:
    """Test the UniGIN layer."""

    @pytest.fixture
    def UniGIN_layer(self):
        """Return a UniGIN layer."""
        in_channels = 10
        intermediate_channels = 20
        out_channels = 30
        nn = torch.nn.Sequential(
            torch.nn.Linear(in_channels, intermediate_channels),
            torch.nn.ReLU(),
            torch.nn.Linear(intermediate_channels, out_channels),
        )
        return UniGINLayer(nn, in_channels)

    def test_forward(self, UniGIN_layer):
        """Test the forward pass of the UniGIN layer."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        output = UniGIN_layer.forward(x, incidence)
        assert output.shape == (3, 30)
