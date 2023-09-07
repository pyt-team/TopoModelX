"""Test the HyperSAGE class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hypersage import HyperSAGE


class TestHyperSAGE:
    """Test the HyperSAGE."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        incidence = torch.from_numpy(np.random.rand(2, 2)).to_sparse()
        incidence = incidence.float().to(device)
        model = HyperSAGE(
            in_channels=2,
            out_channels=2,
            n_layers=2,
            initialization="xavier_uniform",
        ).to(device)
        x_0 = torch.rand(2, 2)

        x_0 = torch.tensor(x_0).float().to(device)

        y1 = model(x_0, incidence)

        assert y1.shape == torch.Size([])
