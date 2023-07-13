"""Test the HyperGAT layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.hypergat_layer import HyperGATLayer


class TestHyperGATLayer:
    """Test the HyperGAT layer."""

    @pytest.fixture
    def hypergat_layer(self):
        """Return a hypergat layer."""
        in_channels = 10
        out_channels = 30
        return HyperGATLayer(in_channels, out_channels)

    def test_forward(self, hypergat_layer):
        """Test the forward pass of the hypergat layer."""
        x_2 = torch.randn(3, 10)
        incidence_2 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        output = hypergat_layer.forward(x_2, incidence_2)
        assert output.shape == (3, 30)

    def test_forward_with_invalid_input(self, hypergat_layer):
        """Test the forward pass of the hypergat layer with invalid input."""
        x_2 = torch.randn(4, 10)
        incidence_2 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        with pytest.raises(RuntimeError):
            hypergat_layer.forward(x_2, incidence_2)

    def test_reset_parameters(self, hypergat_layer):
        """Test the reset_parameters method of the HyperSAGE layer."""
        hypergat_layer.reset_parameters()
        assert hypergat_layer.weight1.requires_grad
        assert hypergat_layer.weight2.requires_grad
        assert hypergat_layer.att_weight1.requires_grad
        assert hypergat_layer.att_weight2.requires_grad

    def test_update(self, hypergat_layer):
        """Test the update function."""
        inputs = torch.randn(10, 20)
        updated = hypergat_layer.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, 20)
