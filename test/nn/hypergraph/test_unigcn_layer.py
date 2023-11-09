"""Test the UniGCN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unigcn_layer import UniGCNLayer


class TestUniGCNLayer:
    """Tests for UniGCN Layer."""

    def test_forward(self):
        """Test forward pass."""
        x = torch.randn(3, 10)
        layer = UniGCNLayer(10, 30)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        x_0, _ = layer.forward(x, incidence)
        assert x_0.shape == torch.Size([3, 30])

        layer = UniGCNLayer(10, 30, use_bn=True)
        x_0, _ = layer.forward(x, incidence)
        assert x_0.shape == torch.Size([3, 30])

    def test_sum_aggregator(self):
        """Test sum aggregator."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        layer = UniGCNLayer(10, 30)
        x_0, _ = layer(x, incidence)
        assert x_0.shape == (3, 30)

    def test_aggregator_validation(self):
        """Test validation."""
        x = torch.randn(10, 20)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)

        layer = UniGCNLayer(10, 30)
        with pytest.raises(Exception) as exc_info:
            layer(x, incidence)
        assert (
            str(exc_info.value)
            == "Mismatch in number of nodes in features and nodes: 10 and 3."
        )
        assert exc_info.type is ValueError

    def test_reset_params(
        self,
    ):
        """Test reset parameters."""
        layer = UniGCNLayer(10, 30)
        layer.conv_level2_1_to_0.weight.requires_grad = False
        layer.conv_level2_1_to_0.weight.fill_(0)
        layer.reset_parameters()
        assert torch.max(layer.conv_level2_1_to_0.weight) > 0

        layer = UniGCNLayer(10, 30, use_bn=True)
        layer.conv_level2_1_to_0.weight.requires_grad = False
        layer.conv_level2_1_to_0.weight.fill_(0)
        layer.reset_parameters()
        assert torch.max(layer.conv_level2_1_to_0.weight) > 0

    def test_aggr_norm(
        self,
    ):
        """Test reset parameters."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)

        layer = UniGCNLayer(10, 30, aggr_norm=True)
        x_0, _ = layer(x, incidence)
        assert x_0.shape == (3, 30)

    def test_batchnorm(self):
        """Test batchnorm."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        layer = UniGCNLayer(10, 30, use_bn=True)
        layer(x, incidence)
        assert layer.bn is not None
        assert layer.bn.num_batches_tracked.item() == 1
