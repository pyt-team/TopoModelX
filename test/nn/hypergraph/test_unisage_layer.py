"""Test the UniSAGE layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unisage_layer import UniSAGELayer


class TestUniSAGELayer:
    """Tests for UniSAGE Layer."""

    @pytest.fixture
    def unisage_layer(self):
        """Fixture for uniSAGE layer."""
        in_channels = 10
        out_channels = 30
        return UniSAGELayer(in_channels, out_channels)

    @pytest.fixture
    def unisage_layer2(self):
        """Fixture for uniSAGE layer."""
        in_channels = 10
        out_channels = 30
        return UniSAGELayer(in_channels, out_channels, use_norm=True)

    def test_forward(self, unisage_layer, unisage_layer2):
        """Test forward pass."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        x_0, x_1 = unisage_layer.forward(x, incidence)
        assert x_0.shape == torch.Size([3, 30])
        assert x_1.shape == torch.Size([3, 30])
        x_0, x_1 = unisage_layer2.forward(x, incidence)
        assert x_0.shape == torch.Size([3, 30])
        assert x_1.shape == torch.Size([3, 30])

    def test_sum_aggregator(self):
        """Test sum aggregator."""
        x = torch.randn(3, 10)
        incidence = torch.tensor([[1, 1, 0], [1, 1, 1], [0, 1, 1]], dtype=torch.float32)
        layer = UniSAGELayer(10, 30, e_aggr="sum")
        x_0, x_1 = layer(x, incidence)

        assert x_0.shape == torch.Size([3, 30])
        assert x_1.shape == torch.Size([3, 30])

    def test_aggregator_validation(self, unisage_layer):
        """Test aggregator validation."""
        with pytest.raises(Exception) as exc_info:
            _ = UniSAGELayer(10, 30, e_aggr="invalid_aggregator")
        assert (
            str(exc_info.value)
            == "Unsupported aggregator: invalid_aggregator, should be 'sum', 'mean',"
        )

    def test_reset_params(self, unisage_layer):
        """Test reset parameters."""
        unisage_layer.linear.weight.requires_grad = False
        unisage_layer.linear.weight.fill_(0)
        unisage_layer.reset_parameters()
        assert torch.max(unisage_layer.linear.weight) > 0
