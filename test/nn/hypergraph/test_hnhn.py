"""Test the HNHN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hnhn import HNHN


class TestHNHN:
    """Test the HNHN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adjacency_1 = torch.from_numpy(np.random.rand(2, 2)).to_sparse()
        adjacency_1 = adjacency_1.float()
        hidden_channels = 5

        model = HNHN(
            in_channels=2,
            hidden_channels=hidden_channels,
            incidence_1=adjacency_1,
            n_layers=2,
        ).to(device)

        x_0 = torch.rand(2, 2).float().to(device)

        x_0, x_1 = model(x_0)
        assert x_0.shape == torch.Size([2, hidden_channels])
        assert x_1.shape == torch.Size([2, hidden_channels])
