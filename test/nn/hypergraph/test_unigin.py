"""Test the UniGIN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.unigin import UniGIN


class TestUniGIN:
    """Test the UniGIN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        n_nodes, n_edges = 2, 5
        incidence = torch.from_numpy(np.random.rand(n_nodes, n_edges)).to_sparse()
        incidence = incidence.float().to(device)
        in_channels, hidden_channels = 2, 10
        model = UniGIN(
            in_channels=in_channels, hidden_channels=hidden_channels, n_layers=2
        ).to(device)
        x_0 = torch.rand(n_nodes, in_channels).float().to(device)
        x_0, x_1 = model(x_0, incidence)

        assert x_0.shape == torch.Size([n_nodes, hidden_channels])
        assert x_1.shape == torch.Size([n_edges, hidden_channels])
