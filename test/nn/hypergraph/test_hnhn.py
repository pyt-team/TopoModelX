"""Test the HNHN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hnhn import HNHN


class TestHNHN:
    """Test the HNHN."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        adjacency_1 = torch.from_numpy(np.random.rand(2, 2)).to_sparse()
        adjacency_1 = adjacency_1.float().to(device)

        model = HNHN(
            channels_node=2,
            channels_edge=2,
            incidence_1=adjacency_1,
            n_classes=1,
            n_layers=2,
        ).to(device)

        x_0 = torch.rand(2, 2)
        x_1 = torch.rand(2, 2)

        x_0, x_1 = (
            torch.tensor(x_0).float().to(device),
            torch.tensor(x_1).float().to(device),
        )

        y1, y2 = model(x_0, x_1)
        assert y1.shape == torch.Size([2, 1])
        assert y2.shape == torch.Size([2])
