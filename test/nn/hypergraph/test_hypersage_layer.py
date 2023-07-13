"""Test the HyperSAGE layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.hypersage_layer import HyperSAGELayer


class TestHyperSAGELayer:
    """Test the HyperSAGE layer."""

    @pytest.fixture
    def hypersage_layer(self):
        """Return a HyperSAGE layer."""
        in_channels = 10
        out_channels = 30
        return HyperSAGELayer(in_channels, out_channels)

    def test_forward(self, hypersage_layer):
        """Test the forward pass of the HyperSAGE layer."""
        x_2 = torch.randn(3, 10)
        incidence_2 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        output = hypersage_layer.forward(x_2, incidence_2)
        assert output.shape == (3, 30)

    def test_forward_with_invalid_input(self, hypersage_layer):
        """Test the forward pass of the HyperSAGE layer with invalid input."""
        x_0 = torch.randn(4, 10)
        incidence_1 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        with pytest.raises(ValueError):
            hypersage_layer.forward(x_0, incidence_1)

    def test_reset_parameters(self, hypersage_layer):
        """Test the reset_parameters method of the HyperSAGE layer."""
        hypersage_layer.reset_parameters()
        assert hypersage_layer.weight.requires_grad