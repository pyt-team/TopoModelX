"""Test the UniGCNII layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unigcnii_layer import UniGCNIILayer


class TestUniGCNIILayer:
    """Test the uniGCNII layer."""

    @pytest.fixture
    def unigcnii_layer(self):
        """Return a uniGCNII layer."""
        in_channels = 10
        alpha = 0.1
        beta = 0.1
        return UniGCNIILayer(
            in_channels = in_channels,
            hidden_channels = in_channels, 
            alpha = alpha, 
            beta = beta
        )

    def test_forward(self, unigcnii_layer):
        """Test the forward pass."""

        n_nodes, in_channels = 3, 10
        x_0 = torch.randn(n_nodes, in_channels)
        incidence_1 = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        x_0, _ = unigcnii_layer.forward(x_0, incidence_1)

        assert x_0.shape == torch.Size([n_nodes, in_channels])

    def test_forward_with_skip(self):
        """Test the forward pass where alpha=1 and beta=0.

        The result should be the same as the skip connection.
        """
        n_nodes, in_channels = 3, 10
        
        x_0 = torch.rand(n_nodes, in_channels).float()
        incidence_1 = torch.tensor([[1, 0], [1, 1], [0, 1]], dtype=torch.float32)
        x_skip =torch.rand(n_nodes, in_channels).float()

        layer = UniGCNIILayer(
            in_channels = in_channels,
            hidden_channels = in_channels, 
            alpha = 1, 
            beta = 0
        )

        x_0, _ = layer(x_0, incidence_1, x_skip)

        torch.testing.assert_close(x_0, x_skip, rtol=1e-4, atol=1e-4)
