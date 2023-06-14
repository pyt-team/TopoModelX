"""Test the template layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.allset_layer import AllSetLayer


class TestAllSetLayer:
    """Test the template layer."""

    @pytest.fixture
    def allset_layer(self):
        """Return a template layer."""
        in_dim = 10
        hid_dim = 20
        out_him = 30
        return TestAllSetLayer(in_dim, hid_dim, out_him)

    def test_forward(self, allset_layer):
        """Test the forward pass of the template layer."""
        x_0 = torch.randn(3, 10)
        incidence_1 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        output = allset_layer.forward(x_0, incidence_1)
        assert output.shape == (3, 30)

    def test_forward_with_invalid_input(self, allset_layer):
        """Test the forward pass of the template layer with invalid input."""
        x_0 = torch.randn(4, 10)
        incidence_1 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        with pytest.raises(ValueError):
            allset_layer.forward(x_0, incidence_1)
