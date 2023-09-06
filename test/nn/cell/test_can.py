"""Test for the CAN class."""

import torch

from topomodelx.nn.cell.can import CAN


class TestCAN:
    """Test CAN."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CAN(
            in_channels_0=2,
            in_channels_1=2,
            out_channels=2,
            dropout=0.5,
            heads=2,
            num_classes=1,
            n_layers=2,
            att_lift=True,
        )

        x_0 = torch.rand(2, 2)
        x_1 = torch.rand(2, 2)

        adjacency_1 = torch.rand(2, 2)
        adjacency_2 = torch.rand(2, 2)
        incidence_2 = torch.rand(2, 2)

        x_0, x_1 = (
            torch.tensor(x_0).float().to(device),
            torch.tensor(x_1).float().to(device),
        )
        adjacency_1 = adjacency_1.float().to(device)
        adjacency_2 = adjacency_2.float().to(device)
        incidence_2 = incidence_2.float().to(device)

        y = model(x_0, x_1, adjacency_1, adjacency_2, incidence_2)
        assert y.shape == torch.Size([1])
