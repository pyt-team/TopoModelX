"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.can_layer import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def test_forward(self):
        """Test the forward method of CANLayer."""
        n_1_cells = 30
        channels = 10

        x_1 = torch.randn(n_1_cells, channels)

        down_laplacian = torch.randn(n_1_cells, n_1_cells).to_sparse().float()
        up_laplacian = torch.randn(n_1_cells, n_1_cells).to_sparse().float()

        # Without attention
        can_layer = CANLayer(
            channels=channels,
            att=False,
        )
        x_1 = can_layer.forward(x_1, down_laplacian, up_laplacian)

        assert x_1.shape == (n_1_cells, channels)

        # With attention
        can_layer = CANLayer(
            channels=channels,
            att=True,
        )
        x_1 = can_layer.forward(x_1, down_laplacian, up_laplacian)
        assert x_1.shape == (n_1_cells, channels)
