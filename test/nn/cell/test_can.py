"""Test for the CAN class."""

import numpy as np
import torch

from topomodelx.nn.cell.can import CAN


class TestCAN:
    """Test CAN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CAN(
            in_channels_0=2,
            in_channels_1=2,
            out_channels=2,
            dropout=0.5,
            heads=2,
            n_layers=2,
            att_lift=False,
            pooling=True,
        ).to(device)

        x_0 = torch.rand(2, 2)
        x_1 = torch.rand(2, 2)

        adjacency_1 = torch.from_numpy(
            np.random.default_rng().random((2, 2))
        ).to_sparse()

        x_0, x_1 = (
            x_0.clone().detach().float().to(device),
            x_1.clone().detach().float().to(device),
        )
        adjacency_1 = adjacency_1.float().to(device)
        adjacency_2 = adjacency_1.float().to(device)
        incidence_2 = adjacency_1.float().to(device)

        x_1 = model(x_0, x_1, adjacency_1, adjacency_2, incidence_2)
        assert x_1.shape == torch.Size([1, 2])
