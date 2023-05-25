"""Test the message passing module."""
import pytest
import torch

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter


class AttentionMessagePassing(MessagePassing):
    """Class to test message passing with attention between cells of same ranks."""

    def __init__(self, in_channels=None, att=False, initialization="xavier_uniform"):
        super().__init__(att=att, initialization=initialization)
        self.in_channels = in_channels
        if att:
            self.att_weight = torch.nn.Parameter(
                torch.Tensor(
                    2 * in_channels,
                )
            )


class TestMessagePassing:
    """Test the MessagePassing class."""

    def setup_method(self):
        """Make message_passing object."""
        self.mp = MessagePassing()
        self.att_mp_xavier_uniform = AttentionMessagePassing(
            in_channels=2, att=True, initialization="xavier_uniform"
        )
        self.att_mp_xavier_normal = AttentionMessagePassing(
            in_channels=2, att=True, initialization="xavier_normal"
        )

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        gain = 1.0
        with pytest.raises(RuntimeError):
            self.mp.initialization = "invalid"
            self.mp.reset_parameters(gain=gain)

        # Test xavier_uniform initialization
        self.mp.initialization = "xavier_uniform"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

        # Test xavier_normal initialization
        self.mp.initialization = "xavier_normal"
        self.mp.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.mp.reset_parameters(gain=gain)
        assert self.mp.weight.shape == (3, 3)

    def custom_message(self, x_source, x_target=None):
        """Make custom message function."""
        return x_source

    def test_propagate(self):
        """Test propagate."""
        x_source = torch.tensor([[1, 2], [3, 4], [5, 6]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        self.mp.message = self.custom_message.__get__(self.mp)

        result = self.mp.propagate(x_source=x_source, neighborhood=neighborhood)
        assert result.shape == (3, 2)

    def test_propagate_with_attention(self):
        """Test propagate with attention."""
        x_source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        self.att_mp_xavier_uniform.message = self.custom_message.__get__(self.mp)
        result = self.att_mp_xavier_uniform.propagate(x_source, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape

        self.att_mp_xavier_normal.message = self.custom_message.__get__(self.mp)
        result = self.att_mp_xavier_normal.propagate(x_source, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape

    def test_attention_between_cells_of_same_rank(self):
        """Test propagate with attention between cells of different ranks."""
        x_source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 0]]),
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        n_messages = 6

        neighborhood = neighborhood.coalesce()
        target_index_i, source_index_j = neighborhood.indices()
        self.att_mp_xavier_uniform.target_index_i = target_index_i
        self.att_mp_xavier_uniform.source_index_j = source_index_j

        result = self.att_mp_xavier_uniform.attention(x_source)
        assert result.shape == (n_messages,)

    def test_attention_between_cells_of_different_ranks(self):
        """Test propagate with attention between cells of different ranks."""
        x_source = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        x_target = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
        neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1], [0, 1, 2, 1, 2]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(2, 3),
        )
        n_messages = 5

        neighborhood = neighborhood.coalesce()
        target_index_i, source_index_j = neighborhood.indices()
        self.att_mp_xavier_uniform.target_index_i = target_index_i
        self.att_mp_xavier_uniform.source_index_j = source_index_j

        result = self.att_mp_xavier_uniform.attention(x_source, x_target)
        assert result.shape == (n_messages,)

    def test_aggregate(self):
        """Test aggregate."""
        x_message = torch.tensor([[1, 2], [3, 4], [5, 6], [3, 4], [5, 6], [5, 6]])
        self.mp.target_index_i = torch.tensor([0, 0, 0, 1, 1, 2])

        result = self.mp.aggregate(x_message)
        expected = torch.tensor([[9, 12], [8, 10], [5, 6]])
        assert torch.allclose(result, expected)

    def test_forward(self):
        """Test forward."""
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        self.mp.message = self.custom_message.__get__(self.mp)
        result = self.mp.forward(x, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape
