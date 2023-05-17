"""Test the message passing module."""
import pytest
import torch

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter


class TestMessagePassing:
    """Test the MessagePassing class."""

    def setup_method(self, method):
        """Make message_passing object."""
        self.message_passing = MessagePassing()

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        gain = 1.0
        with pytest.raises(RuntimeError):
            self.message_passing.initialization = "invalid"
            self.message_passing.reset_parameters(gain=gain)

        # Test xavier_uniform initialization
        self.message_passing.initialization = "xavier_uniform"
        self.message_passing.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.message_passing.reset_parameters(gain=gain)
        assert self.message_passing.weight.shape == (3, 3)

        # Test xavier_normal initialization
        self.message_passing.initialization = "xavier_normal"
        self.message_passing.weight = torch.nn.Parameter(torch.Tensor(3, 3))
        self.message_passing.reset_parameters(gain=gain)
        assert self.message_passing.weight.shape == (3, 3)

    def test_sparsify_message(self):
        """Test sparsify_message."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])

        # Overwrite message function
        def custom_message(self, x):
            """Make custom message function."""
            return x * 2

        original_message_func = self.message_passing.message
        self.message_passing.message = custom_message.__get__(self.message_passing)
        neighborhood = torch.sparse_coo_tensor(
            torch.tensor([[0, 1, 2], [1, 2, 0]]),
            torch.tensor([0.5, 0.6, 0.7]),
            size=(3, 3),
        )
        self.message_passing.propagate(x, neighborhood)
        self.message_passing.message = original_message_func
        result = self.message_passing.sparsify_message(x)
        expected = x * 2
        assert torch.allclose(result, expected)

    def test_get_x_i(self):
        """Test get_x_i."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        self.message_passing.target_index_i = torch.LongTensor([1, 2, 0])
        result = self.message_passing.get_x_i(x)
        expected = torch.Tensor([[4, 5, 6], [7, 8, 9], [1, 2, 3]])
        assert torch.allclose(result, expected)

    def test_aggregate(self):
        """Test aggregate."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        self.message_passing.target_index_i = torch.LongTensor([1, 2, 0])
        result = self.message_passing.aggregate(x)
        expected = scatter(
            x,
            self.message_passing.target_index_i,
            dim=-2,
            reduce=self.message_passing.aggr_func,
        )
        assert torch.allclose(result, expected)

    def test_update(self):
        """Test update."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        self.message_passing.update_func = "sigmoid"
        result = self.message_passing.update(x)
        expected = torch.sigmoid(x)
        assert torch.allclose(result, expected)

        self.message_passing.update_func = "relu"
        result = self.message_passing.update(x)
        expected = torch.relu(x)
        assert torch.allclose(result, expected)

    def test_forward(self):
        """Test forward."""
        x = torch.Tensor([[[1, 2, 3], [4, 5, 6], [7, 8, 9]]])
        neighborhood = torch.sparse_coo_tensor(
            torch.LongTensor([[0, 1, 2], [2, 0, 1]]),
            torch.Tensor([1, 1, 1]),
            size=(3, 3),
        )
        result = self.message_passing.forward(x, neighborhood)
        expected = self.message_passing.propagate(x, neighborhood)
        assert torch.allclose(result, expected)
