"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.can_layer_bis import CANLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def setup_method(self):
        """Set up the CAN for tests."""
        self.n_1_cells = 30
        self.channels = 10

        self.x_1 = torch.randn(self.n_1_cells, self.channels)

        self.down_laplacian = (
            torch.randn(self.n_1_cells, self.n_1_cells).to_sparse().float()
        )
        self.up_laplacian = (
            torch.randn(self.n_1_cells, self.n_1_cells).to_sparse().float()
        )

        # without attention
        self.can_layer = CANLayer(
            channels=self.channels,
            att=False,
        )

        # With attention
        self.can_layer_with_att = CANLayer(
            channels=self.channels,
            att=True,
        )

    def test_forward(self):
        """Test the forward method of CANLayer."""
        result = self.can_layer.forward(
            self.x_1, self.down_laplacian, self.up_laplacian
        )

        assert result.shape == (self.n_1_cells, self.channels)

        result = self.can_layer_with_att.forward(
            self.x_1, self.down_laplacian, self.up_laplacian
        )
        assert result.shape == (self.n_1_cells, self.channels)

    def test_reset_parameters(self):
        """Test the reset_parameters method of CANLayer with attention."""
        with pytest.raises(RuntimeError):
            self.can_layer_with_att.initialization = "invalid"
            self.can_layer_with_att.reset_parameters()

        # Test xavier_uniform on attention
        self.can_layer_with_att.initialization = "xavier_uniform"
        self.can_layer_with_att.att_weight = torch.nn.Parameter(
            torch.Tensor(self.channels, 1)
        )
        self.can_layer_with_att.reset_parameters()
        assert self.can_layer_with_att.att_weight.shape == (self.channels, 1)

        # Test xavier_normal on attention
        self.can_layer_with_att.initialization = "xavier_normal"
        self.can_layer_with_att.att_weight = torch.nn.Parameter(
            torch.Tensor(self.channels, 1)
        )
        self.can_layer_with_att.reset_parameters()
        assert self.can_layer_with_att.att_weight.shape == (self.channels, 1)
