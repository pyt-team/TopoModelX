"""Test for the CCXN class."""

import torch

from topomodelx.nn.cell.ccxn import CCXN


class TestCCXN:
    """Test CCXN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = CCXN(
            in_channels_0=2,
            in_channels_1=2,
            in_channels_2=2,
            n_layers=2,
            att=False,
        ).to(device)

        x_0 = torch.rand(2, 2, dtype=torch.float32, device=device)
        x_1 = torch.rand(2, 2, dtype=torch.float32, device=device)

        adjacency_1 = torch.rand(2, 2, dtype=torch.float32, device=device)
        incidence_2 = torch.rand(2, 2, dtype=torch.float32, device=device)

        x_0, x_1, x_2 = model(x_0, x_1, adjacency_1, incidence_2)
        assert x_0.shape == torch.Size([2, 2])
        assert x_1.shape == torch.Size([2, 2])
        assert x_2.shape == torch.Size([2, 2])
