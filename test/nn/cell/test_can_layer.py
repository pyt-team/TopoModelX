"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.can_layer import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def test_forward(self):
        """Test the forward method of CANLayer."""

        in_channels = 10
        out_channels = 20

        n_cells = 30

        x_0 = torch.randn(n_cells, in_channels)

        lower_neighborhood = torch.randn(n_cells, n_cells)
        upper_neighborhood = torch.randn(n_cells, n_cells)

        lower_neighborhood = lower_neighborhood.to_sparse().float()
        upper_neighborhood = upper_neighborhood.to_sparse().float()

        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr_func="sum",
            update_func="relu"
        )
        x_1 = can_layer.forward(
            x_0, lower_neighborhood, upper_neighborhood
        )
        assert x_1.shape == (n_cells, out_channels)
