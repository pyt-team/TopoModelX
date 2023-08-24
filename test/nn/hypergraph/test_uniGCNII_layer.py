"""Test the UniGCNII layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unigcnii_layer import UniGCNIILayer


class TestUniGCNIILayer:
    """Test the uniGCNII layer."""

    @pytest.fixture
    def uniGCNII_layer(self):
        """Return a uniGCNII layer."""
        in_channels = 10
        alpha = 0.1
        beta = 0.1
        return UniGCNIILayer(in_channels, alpha, beta)

    def test_forward(self, uniGCNII_layer):
        """Test the forward pass."""
        x_0 = torch.randn(3, 10)
        incidence_1 = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        output = uniGCNII_layer.forward(x_0, incidence_1)

        assert output.shape == (3, 10)

    def test_forward_with_skip(self):
        """Test the forward pass where alpha=1 and beta=0. The result should be the same as the skip connection."""
        x_0 = torch.randn(3, 10)
        incidence_1 = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        x_skip = torch.randn(3, 10)

        layer = UniGCNIILayer(10, 1, 0)
        output = layer(x_0, incidence_1, x_skip)

        torch.testing.assert_close(output, x_skip, rtol=1e-4, atol=1e-4)
