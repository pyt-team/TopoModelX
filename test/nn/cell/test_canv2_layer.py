"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.canv2_layer import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def test_forward(self):
        """Test the forward method of CANLayer."""
        in_channels = 7
        out_channels = 64
        dropout = 0.5
        heads = 3
        concat = True
        skip_connection = True

        n_cells = 21

        x_1 = torch.randn(n_cells, in_channels)

        lower_neighborhood = torch.randn(n_cells, n_cells)
        upper_neighborhood = torch.randn(n_cells, n_cells)

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
