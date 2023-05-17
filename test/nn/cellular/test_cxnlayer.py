"""Unit tests for the CXNLayer class."""
import pytest
import torch
from scipy.sparse import coo_matrix

from topomodelx.nn.cellular.cxn_layer import CXNLayer


class TestCXNLayer:
    """Unit tests for the CXNLayer class."""

    @pytest.fixture
    def cxn_layer(self):
        """Fixture to create an instance of CXNLayer for testing."""
        channels = 10
        num_classes = 5
        return CXNLayer(channels, num_classes)

    def test_forward(self):
        """Test the forward method of CXNLayer."""
        n_0_cells = 10
        n_1_cells = 20
        n_2_cells = 30
        channels = 10
        num_classes = 5

        x_0 = torch.randn(n_0_cells, channels)
        x_a_1 = torch.randn(n_1_cells, channels)
        x_b_1 = torch.randn(n_1_cells, channels)
        neighborhood_0_to_0 = torch.randn(
            n_0_cells, n_0_cells
        )  # Neighborhood matrix as torch.Tensor
        neighborhood_1_to_2 = torch.randn(
            n_2_cells, n_1_cells
        )  # Neighborhood matrix as torch.Tensor

        cxn_layer = CXNLayer(channels, num_classes)
        output = cxn_layer.forward(
            x_0, x_a_1, x_b_1, neighborhood_0_to_0, neighborhood_1_to_2
        )

        assert output.shape == (num_classes,)
