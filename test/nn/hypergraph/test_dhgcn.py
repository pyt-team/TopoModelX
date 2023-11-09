"""Test the DHGCNL class."""

import torch

from topomodelx.nn.hypergraph.dhgcn import DHGCN


class TestDHGCNL:
    """Test the DHGCN."""

    def test_forward(self):
        """Test forward method."""
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels, hidden_channels = 8, 4
        model = DHGCN(
            in_channels=in_channels, hidden_channels=hidden_channels, n_layers=2
        )

        n_nodes = 8
        x_0 = torch.rand(n_nodes, in_channels)

        x_0, _ = model(x_0)
        assert x_0.shape == torch.Size([n_nodes, hidden_channels])
