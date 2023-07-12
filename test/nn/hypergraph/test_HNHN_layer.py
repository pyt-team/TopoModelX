"""Test the HNHN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.HNHN_layer import HNHNLayer

torch.manual_seed(41)


class TestHNHNLayer:
    """Test the HNHN layer."""

    @pytest.fixture
    def incidence_1(self):
        """Provide a random incidence matrix of 100 nodes and 20 hyperedges."""
        in_features = 2
        incidence_1 = torch.randint(0, in_features, (100, 20))
        return incidence_1.to_sparse_coo()

    def test_constructor(self, incidence_1):
        """Test the layer constructor in which the weight matrices are computed."""
        in_features = 2
        layer = HNHNLayer(in_features, incidence_1)
        assert layer.weighted_node_to_hyperedge_incidence.shape == (20, 100)
        assert torch.allclose(
            layer.weighted_node_to_hyperedge_incidence.sum(dim=1).to_dense(),
            torch.tensor(1.0),
        )

        assert layer.weighted_hyperedge_to_node_incidence.shape == (100, 20)
        assert torch.allclose(
            layer.weighted_hyperedge_to_node_incidence.sum(dim=1).to_dense(),
            torch.tensor(1.0),
        )

    def test_forward(self, incidence_1):
        """Test the forward pass of the HNHN layer."""
        in_features = 2
        HNHN_layer = HNHNLayer(in_features, incidence_1)
        x_0 = torch.randn(100, in_features)

        x_0, x_1 = HNHN_layer(x_0)
        assert x_0.shape == (100, in_features)
        assert x_1.shape == (20, in_features)
