"""Test the message passing module."""
import pytest
import torch

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter


class AttentionSameRankMP(MessagePassing):
    """Custom class that inherits from MessagePassing to define attention."""

    def __init__(self, in_channels=None, att=False, initialization="xavier_uniform"):
        super().__init__(att=att, initialization=initialization)
        self.in_channels = in_channels
        if att:
            self.att_weight = torch.nn.Parameter(
                torch.Tensor(
                    2 * in_channels,
                )
            )


class AttentionDifferentRanksMP(MessagePassing):
    """Custom class that inherits from MessagePassing to define attention."""

    def __init__(self, in_channels=None, att=False, initialization="xavier_uniform"):
        super().__init__(att=att, initialization=initialization)
        self.in_channels = in_channels
        if att:
            self.att_weight = torch.nn.Parameter(
                torch.Tensor(
                    2 * in_channels,
                )
            )

    def attention(self, x_r, x_s):
        """Compute attention weights for messages between cells of different ranks."""
        return self.attention_between_cells_of_different_ranks(x_r, x_s)


class TestMessagePassing:
    """Test the MessagePassing class."""

    def setup_method(self, method):
        """Make message_passing object."""
        self.mp = MessagePassing()
        self.att_mp_xavier_uniform = AttentionSameRankMP(
            in_channels=2, att=True, initialization="xavier_uniform"
        )
        self.att_mp_xavier_normal = AttentionSameRankMP(
            in_channels=2, att=True, initialization="xavier_normal"
        )
        self.att_mp_different_ranks = AttentionDifferentRanksMP(
            in_channels=2, att=True, initialization="xavier_uniform"
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

    def custom_message(self, x):
        """Make custom message function."""
        return x

    def test_propagate(self):
        """Test propagate."""
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        self.mp.message = self.custom_message.__get__(self.mp)
        result = self.mp.propagate(x, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape

    def test_propagate_with_attention(self):
        """Test propagate with attention."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        self.att_mp_xavier_uniform.message = self.custom_message.__get__(self.mp)
        result = self.att_mp_xavier_uniform.propagate(x, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape

        self.att_mp_xavier_normal.message = self.custom_message.__get__(self.mp)
        result = self.att_mp_xavier_normal.propagate(x, neighborhood)
        expected_shape = (3, 2)
        assert result.shape == expected_shape

    def test_attention_between_cells_of_same_rank(self):
        """Test propagate with attention between cells of different ranks."""
        x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
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

        result = self.att_mp_xavier_uniform.attention_between_cells_of_same_rank(x)
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
        self.att_mp_different_ranks.target_index_i = target_index_i
        self.att_mp_different_ranks.source_index_j = source_index_j

        result = self.att_mp_different_ranks.attention_between_cells_of_different_ranks(
            x_source, x_target
        )
        assert result.shape == (n_messages,)

    def test_sparsify_message(self):
        """Test sparsify_message."""
        x = torch.tensor(
            [
                [
                    1,
                    2,
                ],
                [3, 4],
                [5, 6],
            ]
        )
        neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            values=torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        n_messages = 6

        self.mp.message = self.custom_message.__get__(self.mp)
        _ = self.mp.propagate(x, neighborhood)
        x_sparse = self.mp.sparsify_message(x)
        expected = torch.tensor([[1, 2], [3, 4], [5, 6], [3, 4], [5, 6], [5, 6]])
        assert expected.shape == (n_messages, 2)
        assert torch.allclose(x_sparse, expected)

    def test_get_x_i(self):
        """Test get_x_i."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        self.mp.target_index_i = torch.LongTensor([1, 2, 0])
        result = self.mp.get_x_i(x)
        expected = torch.Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
        assert torch.allclose(result, expected)

    def test_aggregate(self):
        """Test aggregate."""
        x = torch.tensor([[1, 2], [3, 4], [5, 6]])
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 0, 0, 1, 1, 2], [0, 1, 2, 1, 2, 2]]),
            torch.tensor([1, 2, 3, 4, 5, 6]),
            size=(3, 3),
        )
        neighborhood_values = neighborhood.coalesce().values()
        self.mp.message = self.custom_message.__get__(self.mp)
        _ = self.mp.propagate(x, neighborhood)
        x = self.mp.sparsify_message(x)
        x = neighborhood_values.view(-1, 1) * x
        result = self.mp.aggregate(x)
        expected = torch.tensor([[22, 28], [37, 46], [30, 36]])
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
