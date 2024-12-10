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
            n_layers=2,
        ).to(device)

        x_0 = torch.rand(2, 2, dtype=torch.float32, device=device)
        x_1 = torch.rand(2, 2, dtype=torch.float32, device=device)
        x_2 = torch.rand(2, 2, dtype=torch.float32, device=device)
        adjacency_1 = torch.rand(2, 2, dtype=torch.float32, device=device)
        incidence_2 = torch.rand(2, 2, dtype=torch.float32, device=device)
        incidence_1_t = torch.rand(2, 2, dtype=torch.float32, device=device)

        x_0, x_1, x_2 = model(x_0, x_1, x_2, adjacency_1, incidence_2, incidence_1_t)
        assert x_0.shape == torch.Size([2, 16])
        assert x_1.shape == torch.Size([2, 16])
        assert x_2.shape == torch.Size([2, 16])
