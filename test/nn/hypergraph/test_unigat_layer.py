"""Test the UniGAT layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.unigat_layer import UniGATLayer


class TestUniGATLayer:
    """Test the UniGAT layer."""

    @pytest.fixture
    def unigat_layer(self):
        """Return a template uniGAT layer."""
        in_channels = 10
        out_channels = 30
        return UniGATLayer(in_channels, out_channels)

    def test_forward(self, unigat_layer):
        """Test the forward pass of the uniGAT layer."""
        x_0 = torch.randn(3, 10)  # [nodes, in_channels]
        incidence_1 = torch.tensor(
            [[1, 0], [1, 1], [0, 1]], dtype=torch.float32
        ).to_sparse_coo()  # [nodes, edges]

        output = unigat_layer.forward(x_0, incidence_1)
        assert output.shape == (3, 30)  # should be [nodes, out_channels]
