"""Unit tests for the attentional lift layer."""

import pytest
import torch

from topomodelx.nn.cell.attentional_lift_layer import MultiHeadLiftLayer as ALLayer


class TestAttentionalLiftLayer:
    """Unit tests for the Attentional Lift class."""

    def test_forward(self):
        """Test the forward method of Attentional Lift."""
        in_channels = 7
        dropout = 0.5
        heads = 3
        signal_lift_readout = "cat"
        signal_lift_activation = torch.nn.ReLU()

        n_cells = 21
        N = n_cells * n_cells

        x_0 = torch.randn(n_cells, in_channels)

        neighborhood = torch.randn(n_cells, n_cells)
        neighborhood = neighborhood.to_sparse().float()

        can_layer = ALLayer(
            in_channels_0=in_channels,
            heads=3,
            signal_lift_activation=signal_lift_activation,
            signal_lift_dropout=dropout,
            signal_lift_readout=signal_lift_readout,
        )
        x_1 = can_layer.forward(x_0, neighborhood, None)
        if signal_lift_readout == "cat":
            assert x_1.shape == (N, heads)
        else:
            assert x_1.shape == (N, 1)

    def test_reset_parameters(self):
        """Test the reset_parameters method of Attentional Lift."""
        in_channels = 2

        can_layer = ALLayer(
            in_channels_0=in_channels,
        )
        can_layer.reset_parameters()

        for module in can_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
