"""Test the message passing module."""

import torch

from topomodelx.base.message_passing import MessagePassing


class TestMessagePassing:
    """Test the message passing module."""

    def test_init(self):
        """Test the initialization of the message passing module."""
        in_channels = 3
        out_channels = 5
        update_func = "relu"
        initialization = "xavier_uniform"
        mp = MessagePassing(
            in_channels, out_channels, update_func, initialization
        )

        assert mp.in_channels == in_channels
        assert mp.out_channels == out_channels
        assert mp.update_func == update_func
        assert mp.initialization == initialization

    def test_weights(self):
        """Test the weights."""
        in_channels = 3
        out_channels = 5
        update_func = "relu"
        initialization = "xavier_uniform"
        mp = MessagePassing(
            in_channels, out_channels, update_func, initialization
        )

        weight = mp.weight
        assert weight.shape == (in_channels, out_channels)
        assert weight.requires_grad

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        in_channels = 3
        out_channels = 5
        update_func = "relu"
        initialization = "xavier_uniform"
        mp = MessagePassing(
            in_channels, out_channels, update_func, initialization
        )

        weight = mp.reset_parameters()
        assert torch.is_tensor(weight)
        assert weight.requires_grad
        assert weight.shape == (in_channels, out_channels)

    def test_update(self):
        """Test the update function."""
        in_channels = 3
        out_channels = 5
        update_func = "sigmoid"
        initialization = "xavier_uniform"
        mp = MessagePassing(
            in_channels, out_channels, update_func, initialization
        )

        inputs = torch.randn(10, out_channels)
        updated = mp.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, out_channels)

    def test_forward(self):
        """Test the forward pass of the message passing module."""
        in_channels = 3
        out_channels = 5
        n_cells = 10
        update_func = "relu"
        initialization = "xavier_uniform"
        mp = MessagePassing(
            in_channels, out_channels, update_func, initialization
        )

        x = torch.randn(n_cells, in_channels)
        neighborhood = torch.randint(0, 2, (n_cells, n_cells)).float()

        out = mp.forward(x, neighborhood)
        assert torch.is_tensor(out)
        assert out.shape == (10, out_channels)
