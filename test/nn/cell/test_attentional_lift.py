"""Unit tests for the attentional lift layer."""

import pytest
import torch

from topomodelx.nn.cell.attentional_lift_layer import MultiHeadLiftLayer as ALLayer


class TestAttentionalLiftLayer:
    """Unit tests for the Attentional Lift class."""

    def test_forward(self):
        """Test the forward method of Attentional Lift."""
        in_channels_0 = 7
        in_channels_1 = 3
        dropout = 0.5
        heads = 3
        signal_lift_readout = "cat"
        signal_lift_activation = torch.nn.ReLU()

        n_nodes = 3
        n_edges = n_nodes * n_nodes

        x_0 = torch.randn(n_nodes, in_channels_0)
        x_1 = torch.randn(n_edges, in_channels_1)

        neighborhood = torch.randn(n_nodes, n_nodes)
        neighborhood = neighborhood.to_sparse().float()

        can_layer = ALLayer(
            in_channels_0=in_channels_0,
            heads=heads,
            signal_lift_activation=signal_lift_activation,
            signal_lift_dropout=dropout,
            signal_lift_readout=signal_lift_readout,
        )
        x_out = can_layer.forward(x_0, neighborhood, x_1)
        if x_1 is None:
            if signal_lift_readout == "cat":
                assert x_out.shape == (n_edges, heads)
            else:
                assert x_out.shape == (n_edges, 1)
        else:
            if signal_lift_readout == "cat":
                assert x_out.shape == (n_edges, heads + in_channels_1)
            else:
                assert x_out.shape == (n_edges, 1 + in_channels_1)

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
