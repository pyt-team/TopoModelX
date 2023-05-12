"""Test the convolutional layers in the base module."""

import torch

from topomodelx.base.conv import MessagePassingConv


class TestMessagePassingConv:
    """Test the MessagePassingConv class."""

    def test_message_passing_conv_forward(self):
        """Test the forward pass of the message passing convolution layer."""
        in_channels = 3
        out_channels = 5
        n_cells = 10

        # Create a random input tensor
        x = torch.randn((n_cells, in_channels))

        # Create a random neighborhood matrix (adjacency matrix)
        neighborhood = torch.randint(0, 2, (n_cells, n_cells)).float()

        # Create a message passing convolution layer
        mp_conv = MessagePassingConv(
            in_channels=in_channels,
            out_channels=out_channels,
            neighborhood=neighborhood,
            inter_agg_norm=True,
            update_on_message="sigmoid",
            initialization="xavier_uniform",
        )

        # Perform the forward pass
        output = mp_conv(x)

        # Check that the output has the correct shape
        expected_shape = (n_cells, out_channels)
        assert output.shape == expected_shape

        # Check that the output values are within a reasonable range
        assert torch.all(output >= -1.0) and torch.all(output <= 1.0)
