"""Test the convolutional layers in the base module."""

import torch

from topomodelx.base.conv import Conv


class TestConv:
    """Test the Conv class."""

    def setup_method(self):
        """Set up the test."""
        self.in_channels = 3
        self.out_channels = 5
        self.conv = Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            aggr_norm=True,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )

        self.n_cells = 10

        # Create random neighborhood matrix (adjacency matrix)
        self.neighborhood = (
            torch.randint(0, 2, (self.n_cells, self.n_cells)).float().to_sparse()
        )

    def test_update(self):
        """Test the update function."""
        inputs = torch.randn(10, self.out_channels)
        updated = self.conv.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, self.out_channels)

    def test_forward(self):
        """Test the forward pass of the message passing convolution layer."""
        x_source = torch.randn((self.n_cells, self.in_channels))

        output = self.conv.forward(x_source, self.neighborhood)

        assert output.shape == (self.n_cells, self.out_channels)
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
