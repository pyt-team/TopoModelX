"""Test the HMPNN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hmpnn import HMPNN


class TestHMPNN:
    """Test the HMPNN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = 8
        hidden_channels = 32
        model = HMPNN(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
        ).to(device)

        n_nodes, n_edges = 8, 10
        x_0 = torch.rand(n_nodes, in_channels).float().to(device)
        x_1 = torch.rand(n_edges, in_channels).float().to(device)

        adjacency_1 = torch.from_numpy(np.random.rand(n_nodes, n_edges)).to_sparse()
        adjacency_1 = adjacency_1.float().to(device)

        x_0, x_1 = model(x_0, x_1, adjacency_1)
        assert x_0.shape == torch.Size([n_nodes, hidden_channels])
        assert x_1.shape == torch.Size([n_edges, hidden_channels])
