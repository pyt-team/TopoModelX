import unittest

import numpy as np
import torch

from topomodelx.base.message_passing import MessagePassing


class TestMessagePassing(unittest.TestCase):
    """Test the message passing module."""

    def test_init(self):
        """Test the initialization of the message passing module."""
        mp = MessagePassing()
        print(mp)

    def test_forward(self):
        """Test the weights."""
        a = np.array([[-1.0, 0.0], [1.0, -1], [0.0, 1.0]])
        mp = MessagePassing()
        x = torch.rand(2, 10)
        x_out = mp(x, a)
        expected = np.matmul(a, x.numpy())
        np.testing.assert_array_almost_equal(x_out.numpy(), expected)

    def test__propagate(self):
        """Test the _propagate function."""
        a = np.array([[-1.0, 0.0], [1.0, -1], [0.0, 1.0]])
        a_ = np.array([[-1.0, 0.0], [1.0, -0.5], [0.0, 1.0]])
        mp = MessagePassing()
        x = torch.rand(2, 10)
        x_out = mp._propagate(x, a)
        expected = np.matmul(a, x.numpy())
        np.testing.assert_array_almost_equal(x_out.numpy(), expected)

        # test without sign
        x_out = mp._propagate(x, a, aggregate_sign=False)
        expected = np.matmul(abs(a), x.numpy())
        np.testing.assert_array_almost_equal(x_out.numpy(), expected)

        # test without value
        x_out = mp._propagate(x, a_, aggregate_value=False)
        expected = np.matmul(a, x.numpy())
        np.testing.assert_array_almost_equal(x_out.numpy(), expected)

        # test without value and without sign
        x_out = mp._propagate(
            x, a_, aggregate_sign=False, aggregate_value=False
        )  # vaules of a_ will be ignored
        expected = np.matmul(abs(a), x.numpy())
        np.testing.assert_array_almost_equal(x_out.numpy(), expected)

    def test_propagate(self):
        """Test the propagate function."""
        a = np.array([[-1.0, 0.0], [1.0, -1.0], [0.0, 1.0]])
        b = np.array([[1.0, 1.0], [1.0, 1.0], [0.0, 1.0]])
        mp = MessagePassing()
        x = torch.rand(2, 10)
        x_out = mp.propagate(x, [a, b])
        expected_a = np.matmul(a, x.numpy())
        expected_b = np.matmul(b, x.numpy())
        np.testing.assert_array_almost_equal(x_out[0].numpy(), expected_a)
        np.testing.assert_array_almost_equal(x_out[1].numpy(), expected_b)

    def test_update(self):
        """Test the update function."""
        pass

    def test_message(self):
        """Test the message function."""
        pass

    def test_aggregate(self):
        """Test the aggregate function."""
        pass

    def test_get_i(self):
        """Test the get_i function."""
        pass

    def test_get_j(self):
        """Test the get_j function."""
        pass


if __name__ == "__main__":
    unittest.main()
