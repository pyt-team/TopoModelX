"""Unit tests for the CWNLayer class."""

import torch

from topomodelx.nn.cell.cwn_layer import (
    CWNLayer,
    _CWNDefaultAggregate,
    _CWNDefaultFirstConv,
    _CWNDefaultSecondConv,
    _CWNDefaultUpdate,
)


class Test_CWNDefaultFirstConv:
    """Unit tests for the _CWNDefaultFirstConv class."""

    def test_forward(self):
        """Test the forward method of _CWNDefaultFirstConv."""
        n_1_cells = 10
        n_2_cells = 20
        in_channels_1 = 10
        in_channels_2 = 14
        out_channels = 6

        x_1 = torch.randn(n_1_cells, in_channels_1)
        x_2 = torch.randn(n_2_cells, in_channels_2)
        neighborhood_1_to_1 = torch.randn(n_1_cells, n_1_cells)
        neighborhood_2_to_1 = torch.randn(n_1_cells, n_2_cells)

        conv = _CWNDefaultFirstConv(in_channels_1, in_channels_2, out_channels)
        x_1 = conv.forward(x_1, x_2, neighborhood_1_to_1, neighborhood_2_to_1)

        assert x_1.shape == (n_1_cells, out_channels)


class Test_CWNDefaultSecondConv:
    """Unit tests for the _CWNDefaultSecondConv class."""

    def test_forward(self):
        """Test the forward method of _CWNDefaultSecondConv."""
        n_0_cells = 10
        n_1_cells = 20
        in_channels_0 = 7
        in_channels_1 = 5
        out_channels = 12

        x_0 = torch.randn(n_0_cells, in_channels_0)
        x_1 = torch.randn(n_1_cells, in_channels_1)
        neighborhood_0_to_1 = torch.randn(n_1_cells, n_0_cells)

        conv = _CWNDefaultSecondConv(in_channels_0, in_channels_1, out_channels)
        x_1 = conv.forward(x_0, x_1, neighborhood_0_to_1)

        assert x_1.shape == (n_1_cells, out_channels)


class Test_CWNDefaultAggregate:
    """Unit tests for the _CWNDefaultAggregate class."""

    def test_forward(self):
        """Test the forward method of _CWNDefaultAggregate."""
        n_cells = 5
        num_channels = 3
        x = torch.randn(n_cells, num_channels)
        y = torch.randn(n_cells, num_channels)

        aggregation = _CWNDefaultAggregate()
        z = aggregation(x, y)

        assert z.shape == (n_cells, num_channels)


class Test_CWNDefaultUpdate:
    """Unit tests for the _CWNDefaultUpdate class."""

    def test_forward(self):
        """Test the forward method of _CWNDefaultUpdate."""
        n_cells = 5
        in_channels = 3
        out_channels = 7
        x = torch.randn(n_cells, in_channels)
        x_prev = torch.randn(n_cells, in_channels)

        update = _CWNDefaultUpdate(in_channels, out_channels)
        z = update.forward(x, x_prev)

        assert z.shape == (n_cells, out_channels)


class TestCWNLayer:
    """Unit tests for the CWNLayer class."""

    def test_forward(self):
        """Test the forward method of CWNLayer."""
        n_0_cells = 10
        n_1_cells = 20
        n_2_cells = 30
        in_channels = 10
        out_channels = 5

        x_0 = torch.randn(n_0_cells, in_channels)
        x_1 = torch.randn(n_1_cells, in_channels)
        x_2 = torch.randn(n_2_cells, in_channels)
        neighborhood_1_to_1 = torch.randn(n_1_cells, n_1_cells)
        neighborhood_2_to_1 = torch.randn(n_1_cells, n_2_cells)
        neighborhood_0_to_1 = torch.randn(n_1_cells, n_0_cells)

        cwn_layer = CWNLayer(
            in_channels_0=in_channels,
            in_channels_1=in_channels,
            in_channels_2=in_channels,
            out_channels=out_channels,
        )
        x_1 = cwn_layer.forward(
            x_0, x_1, x_2, neighborhood_1_to_1, neighborhood_2_to_1, neighborhood_0_to_1
        )

        assert x_1.shape == (n_1_cells, out_channels)
