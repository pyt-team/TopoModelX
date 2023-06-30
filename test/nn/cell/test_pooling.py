"""Unit tests for the attentional pooling layer."""

import pytest
import torch

from topomodelx.nn.cell.attentional_pooling_layer import PoolLayer


# Write the test for the PoolLayer class here
class TestPoolLayer:
    """Unit tests for the PoolLayer class."""

    def test_forward(self):
        """Test the forward method of PoolLayer."""
        k_pool = 0.75
        in_channels_0 = 96
        signal_pool_activation = torch.nn.ReLU()

        # Input
        x_0 = torch.randn(38, in_channels_0)
        lower_neighborhood = torch.randn(38, 38)
        upper_neighborhood = torch.randn(38, 38)

        # Instantiate the PoolLayer
        pool_layer = PoolLayer(
            in_channels_0=in_channels_0,
            k_pool=k_pool,
            signal_pool_activation=signal_pool_activation,
            readout=True,
        )
        out, lower_neighborhood, upper_neighborhood = pool_layer.forward(
            x_0, lower_neighborhood, upper_neighborhood
        )
        assert out.shape == (int(k_pool * x_0.size(0)), in_channels_0)
        assert lower_neighborhood.shape == (
            int(k_pool * x_0.size(0)),
            int(k_pool * x_0.size(0)),
        )
        assert upper_neighborhood.shape == (
            int(k_pool * x_0.size(0)),
            int(k_pool * x_0.size(0)),
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of PoolLayer."""
        k_pool = 0.75
        in_channels_0 = 96
        signal_pool_activation = torch.nn.ReLU()

        # Instantiate the PoolLayer
        pool_layer = PoolLayer(
            in_channels_0=in_channels_0,
            k_pool=k_pool,
            signal_pool_activation=signal_pool_activation,
            readout=True,
        )
        pool_layer.reset_parameters()
        for module in pool_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
