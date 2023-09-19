"""Test the DHGCN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.dhgcn_layer import DHGCNLayer


class TestDHGCNLayer:
    """Test the DHGCN layer."""

    @pytest.fixture
    def dhgcn_layer(self):
        """Return a DHGCN layer."""
        node_channels = 10
        kn = 3
        km = 4
        return DHGCNLayer(
            in_channels=node_channels,
            intermediate_channels=node_channels,
            out_channels=node_channels,
            k_neighbours=kn,
            k_centroids=km,
        )

    def test_kmeans(self, dhgcn_layer):
        """Test the forward pass of the template layer."""
        x_0 = torch.randn(4, 10)
        output = dhgcn_layer.kmeans(x_0, k=dhgcn_layer.k_centroids)
        assert output.shape == (2, 4)
        if torch.cuda.is_available():
            x_0 = x_0.to("cuda")
            output = dhgcn_layer.kmeans(x_0, k=dhgcn_layer.k_centroids)
            assert output.device.type == x_0.device.type

    def test_kmeans_with_invalid_input(self, dhgcn_layer):
        """Test the forward pass of the template layer with invalid input."""
        x_0 = torch.randn(4, 10)
        with pytest.raises(ValueError):
            dhgcn_layer.kmeans(x_0, k=5)

    def test_forward(self, dhgcn_layer):
        """Test the forward pass of the DHGCN layer."""
        x = torch.randn(4, 10)
        output = dhgcn_layer.forward(x)
        assert output.shape == (4, 10)

    def test_forward_invalid_input(self, dhgcn_layer):
        """Test the forward pass of the DHGCN layer."""
        x = torch.randn(2, 10)
        with pytest.raises(ValueError):
            dhgcn_layer.forward(x)
