"""Test the HyperSAGE class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.hypersage import HyperSAGE


class TestHyperSAGE:
    """Test the HyperSAGE."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        incidence = torch.from_numpy(np.random.default_rng().random((2, 2))).to_sparse()
        incidence = incidence.float().to(device)
        model = HyperSAGE(
            in_channels=2,
            hidden_channels=2,
            n_layers=2,
            device=device,
            initialization="xavier_uniform",
        ).to(device)
        x_0 = torch.rand(2, 2).float().to(device)
        x_0 = model(x_0, incidence)

        assert x_0.shape == torch.Size([2, 2])
