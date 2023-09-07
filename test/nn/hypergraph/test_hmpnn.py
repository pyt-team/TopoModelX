"""Test the HMPNN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hmpnn import HMPNN


class TestHMPNN:
    """Test the HMPNN."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HMPNN(8, (8, 8), 1, 1).to(device)

        x_0 = torch.rand(8, 8)
        x_1 = torch.rand(8, 8)

        adjacency_1 = torch.from_numpy(np.random.rand(8, 8)).to_sparse()

        x_0, x_1 = (
            torch.tensor(x_0).float().to(device),
            torch.tensor(x_1).float().to(device),
        )
        adjacency_1 = adjacency_1.float().to(device)

        y = model(x_0, x_1, adjacency_1)
        assert y.shape == torch.Size([8, 1])
