"""Unit tests for the CCXNLayer class."""

import pytest
import torch

from topomodelx.nn.cell.ccxn_layer import CCXNLayer


class TestCCXNLayer:
    """Unit tests for the CCXNLayer class."""

    def test_forward(self):
        """Test the forward method of CCXNLayer."""
        n_0_cells = 10
        n_1_cells = 20
        n_2_cells = 30
        channels = 10

        x_0 = torch.randn(n_0_cells, channels)
        x_1 = torch.randn(n_1_cells, channels)
        x_2 = torch.randn(n_2_cells, channels)
        neighborhood_0_to_0 = torch.randn(n_0_cells, n_0_cells)
        neighborhood_1_to_2 = torch.randn(n_2_cells, n_1_cells)

        # Without attention
        cxn_layer = CCXNLayer(
            in_channels_0=channels,
            in_channels_1=channels,
            in_channels_2=channels,
            att=False,
        )
        x_0, x_1, x_2 = cxn_layer.forward(
            x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2
        )

        assert x_0.shape == (n_0_cells, channels)
        assert x_1.shape == (n_1_cells, channels)
        assert x_2.shape == (n_2_cells, channels)

        # With attention: between x_0 <-> x_0 cells and x_1 <-> x_2 cells
        neighborhood_0_to_0 = neighborhood_0_to_0.to_sparse().float()
        neighborhood_1_to_2 = neighborhood_1_to_2.to_sparse().float()
        cxn_layer = CCXNLayer(
            in_channels_0=channels,
            in_channels_1=channels,
            in_channels_2=channels,
            att=True,
        )
        x_0, x_1, x_2 = cxn_layer.forward(
            x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2, x_2
        )
        assert x_0.shape == (n_0_cells, channels)
        assert x_1.shape == (n_1_cells, channels)
        assert x_2.shape == (n_2_cells, channels)
