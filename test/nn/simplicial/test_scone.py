"""Unit tests for Scone Model."""
import random

import numpy as np
import torch
from torch.utils.data import DataLoader

from topomodelx.nn.simplicial.scone import (
    SCoNe,
    TrajectoriesDataset,
    generate_complex,
    generate_trajectories,
)


class TestScone:
    """Unit tests for the Scone model class."""

    def test_forward(self):
        """Test the forward method of Scone."""
        torch.manual_seed(42)
        random.seed(42)
        N = 150
        sc, coords = generate_complex(N, rng=np.random.default_rng(42))
        incidence_1 = torch.Tensor(sc.incidence_matrix(1).toarray())
        incidence_2 = torch.Tensor(sc.incidence_matrix(2).toarray())

        trajectories = generate_trajectories(sc, coords, 1200)
        dataset = TrajectoriesDataset(sc, trajectories)
        train_dl = DataLoader(dataset, batch_size=1, shuffle=True)
        in_channels = 1
        hidden_channels = 16
        n_layers = 6
        model = SCoNe(in_channels, hidden_channels, n_layers)
        batch = next(iter(train_dl))
        traj, mask, last_nodes = batch
        with torch.no_grad():
            forward_pass = model(traj, incidence_1, incidence_2)
        print(forward_pass[0][0])
        assert torch.any(
            torch.isclose(
                forward_pass[0][0],
                torch.tensor(
                    [
                        1.0000,
                        -0.9999,
                        0.8368,
                        -0.9315,
                        -0.1243,
                        -0.9998,
                        -0.9989,
                        -1.0000,
                        -0.2776,
                        -0.8334,
                        0.6433,
                        -0.9924,
                        -0.9983,
                        -0.9999,
                        -0.9947,
                        -0.6742,
                    ]
                ),
                rtol=1e-02,
            )
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of Scone."""
        in_channels = 1
        hidden_channels = 16
        n_layers = 6
        model = SCoNe(in_channels, hidden_channels, n_layers)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
