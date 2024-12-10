"""Tests for the HMC class."""

import numpy as np
import torch

from topomodelx.nn.combinatorial.hmc import HMC


class TestHMC:
    """Unit tests for the HMC class."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        in_channels = [2, 2, 2]
        intermediate_channels = [2, 2, 2]
        final_channels = [2, 2, 2]
        channels_per_layer = [[in_channels, intermediate_channels, final_channels]]
        model = HMC(channels_per_layer, negative_slope=0.2).to(device)

        x_0 = torch.rand(2, 2, dtype=torch.float32, device=device)
        x_1 = torch.rand(2, 2, dtype=torch.float32, device=device)
        x_2 = torch.rand(2, 2, dtype=torch.float32, device=device)
        adjacency_0 = torch.from_numpy(
            np.random.default_rng().random((2, 2))
        ).to_sparse()

        adjacency_0 = adjacency_0.float().to(device)

        x_0, x_1, x_2 = model(
            x_0,
            x_1,
            x_2,
            adjacency_0,
            adjacency_0,
            adjacency_0,
            adjacency_0,
            adjacency_0,
        )
        assert x_0.shape == torch.Size([2, 2])
        assert x_1.shape == torch.Size([2, 2])
        assert x_2.shape == torch.Size([2, 2])
