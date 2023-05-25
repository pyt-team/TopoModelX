"""Test the convolutional layers in the base module."""

import torch

from topomodelx.base.conv import Conv


class TestConv:
    """Test the Conv class."""

    def test_update(self):
        """Test the update function."""
        in_channels = 3
        out_channels = 5

        conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr_norm=True,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )

        inputs = torch.randn(10, out_channels)
        updated = conv.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, out_channels)

    def test_forward(self):
        """Test the forward pass of the message passing convolution layer."""
        in_channels = 3
        out_channels = 5
        n_cells = 10

        # Create a random input tensor
        x = torch.randn((n_cells, in_channels))

        # Create a random neighborhood matrix (adjacency matrix)
        neighborhood = torch.randint(0, 2, (n_cells, n_cells)).float().to_sparse()

        # Create a message passing convolution layer
        conv = Conv(
            in_channels=in_channels,
            out_channels=out_channels,
            aggr_norm=True,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )

        # Perform the forward pass
        output = conv.forward(x, neighborhood)

        # Check that the output has the correct shape
        expected_shape = (n_cells, out_channels)
        assert output.shape == expected_shape

        # Check that the output values are within a reasonable range
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
