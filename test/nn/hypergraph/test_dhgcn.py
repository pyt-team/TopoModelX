"""Test the DHGCNL class."""

import torch

from topomodelx.nn.hypergraph.dhgcn import DHGCN


class TestDHGCNL:
    """Test the DHGCN."""

    def test_fowared(self):
        """Test forward method."""
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = DHGCN(channels_node=2, n_layers=2)

        x_0 = torch.rand(2, 2)

        y = model(x_0)
        assert y.shape == torch.Size([1])
