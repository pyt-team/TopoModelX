"""Unit tests for the attentional lift layer."""

import itertools

import pytest
import torch

from topomodelx.nn.cell.attentional_lift_layer import MultiHeadLiftLayer


class TestAttentionalLiftLayer:
    """Unit tests for the Attentional Lift class."""

    def test_default_parameters(self):
        """Test the default parameters of Attentional Lift."""
        layer = MultiHeadLiftLayer(in_channels_0=16, heads=32)
        assert layer is not None

    def test_skip_connection_false(self):
        """Test without skip connection."""
        layer = MultiHeadLiftLayer(in_channels_0=16, heads=32, skip_connection=False)
        assert layer is not None

    def test_heads_0(self):
        """Test with heads=0."""
        with pytest.raises(AssertionError):
            MultiHeadLiftLayer(in_channels_0=16, heads=0)

    def test_invalid_readout_method(self):
        """Test with invalid readout method."""
        with pytest.raises(AssertionError):
            MultiHeadLiftLayer(in_channels_0=16, signal_lift_readout="invalid")

    def test_forward(self):
        """Test the forward method of Attentional Lift."""
        in_channels_0 = 7
        in_channels_1 = 3
        dropout = 0.5
        heads = [1,3]
        signal_lift_readout = ["cat", "sum", "avg", "max"]
        signal_lift_activation = torch.nn.ReLU()

        n_nodes = 3
        n_edges = n_nodes * n_nodes

        x_0 = torch.randn(n_nodes, in_channels_0)
        x_1 = torch.randn(n_edges, in_channels_1)

        neighborhood = torch.randn(n_nodes, n_nodes)
        neighborhood = neighborhood.to_sparse().float()

        combinations = itertools.product(heads, signal_lift_readout)

        for head, signal_lift_read in combinations:

            can_layer = MultiHeadLiftLayer(
                in_channels_0=in_channels_0,
                heads=head,
                signal_lift_activation=signal_lift_activation,
                signal_lift_dropout=dropout,
                signal_lift_readout=signal_lift_read,
            )
            x_out = can_layer.forward(x_0, neighborhood, x_1)

            if signal_lift_read == "cat":
                assert x_out.shape == (n_edges, head + in_channels_1)
            else:
                assert x_out.shape == (n_edges, 1 + in_channels_1)

    def test_reset_parameters(self):
        """Test the reset_parameters method of Attentional Lift."""
        in_channels = 2

        can_layer = MultiHeadLiftLayer(
            in_channels_0=in_channels,
        )
        can_layer.reset_parameters()

        for module in can_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
