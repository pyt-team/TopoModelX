"""Test the SAN layer."""
import torch

from topomodelx.nn.simplicial.san_layer import SANLayer


class TestSANLayer:
    """Unit tests for the SANLayer class."""

    def test_forward(self):
        """Test the forward method of SANLayer."""
        in_channels = 2
        out_channels = 5
        num_filters_J = 2

        san_layer = SANLayer(in_channels, out_channels, num_filters_J)

        # Create input tensors
        n_cells = 100
        x = torch.randn(n_cells, in_channels)
        Lup = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            values=torch.tensor([0.5, 0.3, 0.2]),
            size=(n_cells, n_cells),
        )
        Ldown = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2], [1, 2, 0]]),
            values=torch.tensor([0.3, 0.4, 0.5]),
            size=(n_cells, n_cells),
        )
        P = torch.randn(n_cells, n_cells)

        # Perform forward pass
        output = san_layer(x, Lup, Ldown, P)
        assert output.shape == (n_cells, out_channels)

    def test_reset_parameters(self):
        """Test the reset_parameters method of SANLayer."""
        in_channels = 2
        out_channels = 5
        num_filters_J = 2

        san_layer = SANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            num_filters_J=num_filters_J,
        )
        san_layer.reset_parameters()

        for module in san_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
