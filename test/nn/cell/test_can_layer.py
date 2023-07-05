"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.can_layer import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def test_reset_parameters(self):
        """Test the reset_parameters method of CANLayer."""
        in_channels = 2
        out_channels = 5

        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        can_layer.reset_parameters()

        for module in can_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def test_default_parameters(self):
        """Test the default parameters of CANLayer."""
        layer = CANLayer(in_channels=16, out_channels=32)
        assert layer is not None

    def test_skip_connection_false(self):
        """Test without skip connection."""
        layer = CANLayer(in_channels=16, out_channels=32, skip_connection=False)
        assert layer is not None

    def test_concat_false(self):
        """Test without concatenation."""
        layer = CANLayer(in_channels=16, out_channels=32, concat=False)
        assert layer is not None

    def test_heads(self):
        """Test with multiple heads."""
        layer = CANLayer(in_channels=16, out_channels=32, heads=3)
        assert layer is not None

    def test_dropout(self):
        """Test with dropout."""
        layer = CANLayer(in_channels=16, out_channels=32, dropout=0.5)
        assert layer is not None

    def test_in_channels_0(self):
        """Test with in_channels_0."""
        with pytest.raises(ValueError):
            layer = CANLayer(in_channels=0, out_channels=32)

    def test_forward(self):
        """Test the forward method of CANLayer."""
        # set seed for reproducibility
        torch.manual_seed(0)

        in_channels = 7
        out_channels = 2
        dropout = 0.0
        heads = 1
        concat = True
        skip_connection = True

        n_cells = 3

        x_1 = torch.randn(n_cells, in_channels)

        lower_neighborhood = torch.randint(0, 2, (n_cells, n_cells)).float()
        upper_neighborhood = torch.randint(0, 2, (n_cells, n_cells)).float()

        lower_neighborhood = lower_neighborhood.to_sparse().float()
        upper_neighborhood = upper_neighborhood.to_sparse().float()

        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            dropout=dropout,
            heads=heads,
            concat=concat,
            skip_connection=skip_connection,
        )
        x_1 = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
        if concat:
            assert x_1.shape == (n_cells, out_channels * heads)
        else:
            assert x_1.shape == (n_cells, out_channels)
