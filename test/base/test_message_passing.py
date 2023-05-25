"""Test the message passing module."""
import pytest
import torch

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter


def custom_message(x_source, x_target=None):
    """Make custom message function."""
    return x_source


class TestMessagePassing:
    """Test the MessagePassing class."""

    def setup_method(self):
        """Make message_passing object."""
        self.x_source = torch.tensor([[1, 2], [3, 4], [5, 6]]).float()
        self.x_target = torch.tensor([[1, 2], [3, 4]]).float()
        self.neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        ).float()
        self.neighborhood_r_to_s = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(2, 3),
        )

        self.mp = MessagePassing()
        self.mp.message = custom_message

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        gain = 1.0
        with pytest.raises(RuntimeError):
            self.mp.initialization = "invalid"
            self.mp.reset_parameters(gain=gain)

        # Test xavier_uniform
        self.mp.initialization = "xavier_uniform"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

        # Test xavier_normal
        self.mp.initialization = "xavier_normal"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

        # Test with attention weights & xavier_uniform
        self.mp.att = True
        self.mp.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )
        self.mp.initialization = "xavier_uniform"
        self.mp.reset_parameters(gain=gain)
        assert self.mp.att_weight.shape == (4,)

        # Test with attention weights & xavier_normal
        self.mp.att = True
        self.mp.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )
        self.mp.initialization = "xavier_normal"
        self.mp.reset_parameters(gain=gain)
        assert self.mp.att_weight.shape == (4,)

    def test_propagate(self):
        """Test propagate."""
        # Test without attention
        self.mp.att = False
        result = self.mp.propagate(self.x_source, self.neighborhood)
        assert result.shape == (3, 2)

        # Test with attention (source & target on the same cells)
        self.mp.att = True
        self.mp.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )

        result = self.mp.propagate(self.x_source, self.neighborhood)
        assert result.shape == (3, 2)

        # Test with attention (source & target on different cells)
        self.mp.att = True
        self.mp.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )
        result = self.mp.propagate(
            self.x_source, self.neighborhood_r_to_s, self.x_target
        )
        assert result.shape == (2, 2)

    def test_attention(self):
        """Test attention."""
        self.mp.att = True
        self.mp.att_weight = torch.nn.Parameter(
            torch.Tensor(
                2 * 2,
            )
        )

        # Test with source & target on the same cells
        neighborhood = self.neighborhood.coalesce()
        self.mp.target_index_i, self.mp.source_index_j = neighborhood.indices()
        n_messages = len(
            self.mp.target_index_i,
        )
        result = self.mp.attention(self.x_source)
        assert result.shape == (n_messages,)

        # Test with source & target on different cells
        neighborhood_r_to_s = self.neighborhood_r_to_s.coalesce()
        self.mp.target_index_i, self.mp.source_index_j = neighborhood_r_to_s.indices()
        n_messages = len(
            self.mp.target_index_i,
        )
        result = self.mp.attention(self.x_source, self.x_target)
        assert result.shape == (n_messages,)

    def test_aggregate(self):
        """Test aggregate."""
        x_message = torch.tensor([[1, 2], [3, 4], [5, 6], [3, 4], [5, 6], [5, 6]])
        self.mp.target_index_i = torch.tensor([0, 0, 1, 1, 2, 2])

        result = self.mp.aggregate(x_message)
        expected = torch.tensor([[4, 6], [8, 10], [10, 12]])
        assert torch.allclose(result, expected)

    def test_forward(self):
        """Test forward."""
        result = self.mp.forward(self.x_source, self.neighborhood)
        assert result.shape == (3, 2)
