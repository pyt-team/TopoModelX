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
        np.random.seed(42)
        torch.manual_seed(42)
        random.seed(42)
        N = 150
        sc, coords = generate_complex(N)
        trajectories = generate_trajectories(sc, coords, 1200)
        dataset = TrajectoriesDataset(sc, trajectories)
        train_dl = DataLoader(dataset, batch_size=1, shuffle=True)
        hidden_dims = [16, 16, 16, 16, 16, 16]
        model = SCoNe(sc, hidden_dims)
        batch = next(iter(train_dl))
        traj, mask, last_nodes = batch
        with torch.no_grad():
            forward_pass = model(traj, mask)
        assert torch.any(
            torch.isclose(forward_pass[0][0], torch.tensor([-46.2888]), rtol=1e-02)
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of Scone."""
        N = 150
        sc, coords = generate_complex(N)
        hidden_dims = [16, 16, 16, 16, 16, 16]
        model = SCoNe(sc, hidden_dims)
        for layer in model.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        for module in model.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
