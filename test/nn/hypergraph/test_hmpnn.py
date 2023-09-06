"""Test the HMPNN class."""

import torch

from topomodelx.nn.hypergraph.hmpnn import HMPNN


class TestHMPNN:
    """Test the HMPNN."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = HMPNN(8, (8, 8), 1, 1).to(device)

        x_0 = torch.rand(8, 8)

        y = model(x_0)
        assert y.shape == torch.Size([1])
