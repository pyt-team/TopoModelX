"""Unit tests for the CXNLayer class."""

import pytest
import torch

from topomodelx.nn.cell.cxn_layer import CXNLayer


class TestCXNLayer:
    """Unit tests for the CXNLayer class."""

    def test_forward(self):
        """Test the forward method of CXNLayer."""
        n_0_cells = 10
        n_1_cells = 20
        n_2_cells = 30
        channels = 10

        x_0 = torch.randn(n_0_cells, channels)
        x_1 = torch.randn(n_1_cells, channels)
        neighborhood_0_to_0 = torch.randn(n_0_cells, n_0_cells)
        neighborhood_1_to_2 = torch.randn(n_2_cells, n_1_cells)

        # Without attention
        cxn_layer = CXNLayer(
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

        # With attention
        neighborhood_0_to_0 = neighborhood_0_to_0.to_sparse().float()
        neighborhood_1_to_2 = neighborhood_1_to_2.to_sparse().float()
        cxn_layer = CXNLayer(
            cin_channels_0=channels,
            in_channels_1=channels,
            in_channels_2=channels,
            att=True,
        )
        x_0, x_1, x_2 = cxn_layer.forward(
            x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2
        )
        assert x_0.shape == (n_0_cells, channels)
        assert x_1.shape == (n_1_cells, channels)
        assert x_2.shape == (n_2_cells, channels)
