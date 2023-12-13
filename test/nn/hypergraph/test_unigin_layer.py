"""Test the UniGIN layer."""
import numpy as np
import pytest
import torch

from topomodelx.nn.hypergraph.unigin_layer import UniGINLayer


class TestUniGINLayer:
    """Test the UniGIN layer."""

    @pytest.fixture
    def unigin_layer(self):
        """Return a UniGIN layer."""
        self.in_channels = 10
        return UniGINLayer(in_channels=self.in_channels)

    @pytest.fixture
    def unigin_layer2(self):
        """Return a UniGIN layer."""
        self.in_channels = 10
        return UniGINLayer(in_channels=self.in_channels, use_norm=True)

    def test_forward(self, unigin_layer, unigin_layer2):
        """Test the forward pass of the UniGIN layer."""
        n_nodes, n_edges = 2, 3
        incidence = torch.from_numpy(np.random.rand(n_nodes, n_edges)).to_sparse()
        incidence = incidence.float()
        x_0 = torch.rand(n_nodes, self.in_channels).float()
        x_0, x_1 = unigin_layer.forward(x_0, incidence)

        assert x_0.shape == torch.Size([n_nodes, self.in_channels])
        assert x_1.shape == torch.Size([n_edges, self.in_channels])

        x_0, x_1 = unigin_layer2.forward(x_0, incidence)

        assert x_0.shape == torch.Size([n_nodes, self.in_channels])
        assert x_1.shape == torch.Size([n_edges, self.in_channels])
