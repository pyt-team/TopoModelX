"""Test for the CWN class."""

import torch

from topomodelx.nn.cell.cwn import CWN


class TestCWN:
    """Test CWN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CWN(
            in_channels_0=2,
            in_channels_1=2,
            in_channels_2=2,
            hid_channels=16,
            num_classes=1,
            n_layers=2,
        )

        x_0 = torch.rand(2, 2)
        x_1 = torch.rand(2, 2)
        x_2 = torch.rand(2, 2)
        adjacency_1 = torch.rand(2, 2)
        incidence_2 = torch.rand(2, 2)
        incidence_1_t = torch.rand(2, 2)

        x_0, x_1, x_2 = (
            torch.tensor(x_0).float().to(device),
            torch.tensor(x_1).float().to(device),
            torch.tensor(x_2).float().to(device),
        )
        adjacency_1 = adjacency_1.float().to(device)
        incidence_2 = incidence_2.float().to(device)
        incidence_1_t = incidence_1_t.float().to(device)

        y = model(x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t)
        assert y.shape == torch.Size([1])
