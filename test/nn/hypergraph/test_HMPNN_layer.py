"""Test the HMPNNN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.HMPNN_layer import (
    DefaultHyperedgeToNodeMessagingFunc,
    HMPNNLayer,
    HyperedgeToNodeMessanger,
    NodeToHyperedgeMessanger,
)

torch.manual_seed(41)


class TestHMPNNLayer:
    """Test the HMPNN layer."""

    @pytest.fixture
    def incidence_1(self):
        """Provide a random incidence matrix with 100 nodes and 20 hyperedges."""
        in_features = 2
        incidence_1 = torch.randint(0, in_features, (100, 20))
        return incidence_1.to_sparse_coo()

    def test_forward(self, incidence_1):
        """Test the forward pass of the HMPNN layer."""
        in_features = 2
        HMPNN_layer = HMPNNLayer(in_features)
        x_0 = torch.randn(100, in_features)
        x_1 = torch.randn(20, in_features)

        x_0, x_1 = HMPNN_layer(x_0, x_1, incidence_1)
        assert x_0.shape == (100, in_features)
        assert x_1.shape == (20, in_features)

    def test_node_to_hyperedge_messanger(self, incidence_1):
        """Test NodeToHyperedgeMessanger."""
        in_features = 2
        messanger = NodeToHyperedgeMessanger(torch.nn.functional.sigmoid)
        x_0 = torch.randn(100, in_features)
        node_messages_aggregated, node_messages = messanger(x_0, incidence_1)
        assert node_messages_aggregated.shape == (incidence_1.size(1), in_features)
        assert node_messages.shape == (incidence_1.size(0), in_features)

    def test_hyperedge_to_node_messanger(self, incidence_1):
        """Test HyperedgeToNodeMessanger."""
        in_features = 2
        messanger = HyperedgeToNodeMessanger(
            DefaultHyperedgeToNodeMessagingFunc(in_features)
        )
        x_1 = torch.randn(20, in_features)
        node_messages = torch.randn(100, in_features)
        hyperedge_messages_aggregated = messanger(x_1, incidence_1, node_messages)
        assert hyperedge_messages_aggregated.shape == (incidence_1.size(0), in_features)
