"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.can_layer import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

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
        x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)

        # test forward with different number of heads predifined arguments
        heads = 3
        can_layer = CANLayer(
            in_channels=in_channels, out_channels=out_channels, heads=heads
        )
        x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)

        # test forward without skip connection predifined arguments
        skip_connection = False
        heads = 1
        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            skip_connection=skip_connection,
        )
        x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)

        # test forward without concat and predifined arguments
        concat = False
        heads = 1
        skip_connection = True
        can_layer = CANLayer(
            in_channels=in_channels, out_channels=out_channels, concat=concat
        )
        x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)

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
