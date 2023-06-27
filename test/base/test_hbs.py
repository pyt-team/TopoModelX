"""Test the Higher Order Attention Block for squared neighborhoods (HBS) layer in the base module."""
import math

import pytest
import torch

from topomodelx.base.hbs import HBS


class TestHBS:
    """Test the HBS class."""

    def set_weights_to_one(self):
        """Set the weights to constant values."""
        for w, a in zip(self.hbs.weight, self.hbs.att_weight):
            torch.nn.init.constant_(w, 1.0)
            torch.nn.init.constant_(a, 1.0)

    def setup_method(self):
        """Set up the test."""
        self.d_s_in, self.d_s_out = 2, 2
        self.neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        self.hbs = HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=1,
        )

    def test_forward_shape(self):
        """Test the shapes of the outputs of the forward pass of the HBS layer."""
        self.d_s_out = 3
        self.hbs = HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=2,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )
        self.n_source_cells = 10
        self.neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 1, 2, 9], [3, 7, 9, 2, 5]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(10, 10),
            dtype=torch.float,
        )
        """Test the forward pass of the message passing convolution layer."""
        x_source = torch.tensor(
            [
                [1, 2],
                [2, 3],
                [3, 3],
                [4, 4],
                [5, 4],
                [6, 9],
                [7, 3],
                [8, 7],
                [9, 7],
                [10, -1],
            ]
        ).float()

        result = self.hbs.forward(x_source, self.neighborhood)

        assert result.shape == (self.n_source_cells, self.d_s_out)

    def test_initialization(self):
        """Test if the initialization of the parameter weights works."""
        HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=2,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )
        HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=2,
            update_func="sigmoid",
            initialization="xavier_normal",
        )
        with pytest.raises(RuntimeError):
            HBS(
                source_in_channels=self.d_s_in,
                source_out_channels=self.d_s_out,
                negative_slope=0.2,
                softmax=False,
                m_hop=2,
                update_func="sigmoid",
                initialization="non_existing",
            )


    def test_attention_without_softmax(self):
        """Test the attention matrix calculation without softmax."""
        self.set_weights_to_one()
        # Create the message that will be used for the attention.
        message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        # Calculate the attention matrix.
        attention_matrix = self.hbs.attention(
            message, self.neighborhood, self.hbs.att_weight[0]
        ).to_dense()
        # Create the expected attention matrix. The values have been calculated by hand.
        expected_attention_matrix = self.neighborhood.to_dense() * torch.tensor(
            [
                [6.0 / 24.0, 10.0 / 24.0, 14.0 / 24.0],
                [10.0 / 28.0, 14.0 / 28.0, 18.0 / 28.0],
                [14.0 / 32.0, 18.0 / 32.0, 22.0 / 32.0],
            ],
            dtype=torch.float,
        )
        # Compare the two attention matrices.
        assert torch.allclose(attention_matrix, expected_attention_matrix)

    def test_attention_with_softmax(self):
        """Test the attention matrix calculation with softmax."""
        self.hbs.softmax = True
        self.set_weights_to_one()
        # Create the message that will be used for the attention.
        message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        # Calculate the attention matrix.
        attention_matrix = self.hbs.attention(
            message, self.neighborhood, self.hbs.att_weight[0]
        ).to_dense()
        # Create the expected attention matrix. The values have been calculated by hand.
        expected_attention_matrix_wo_softmax_nor_product = torch.tensor(
            [[6.0, 10.0, 14.0], [10.0, 14.0, 18.0], [14.0, 18.0, 22.0]],
            dtype=torch.float,
        )
        expected_attention_wo_product = torch.exp(
            expected_attention_matrix_wo_softmax_nor_product
        )
        row_normalization_denominator = torch.tensor(
            [
                1.0 / (math.exp(10.0) + math.exp(14.0)),
                1.0 / (math.exp(10.0) + math.exp(18.0)),
                1.0 / (math.exp(14.0) + math.exp(18.0)),
            ]
        )
        expected_attention_wo_product = (
            expected_attention_wo_product.T * row_normalization_denominator
        ).T
        expected_attention_matrix = (
            self.neighborhood.to_dense() * expected_attention_wo_product
        )
        # Compare the two attention matrices.
        assert torch.allclose(attention_matrix, expected_attention_matrix)

    def test_forward_values(self):
        """Test the values of the outputs of the forward pass of the HBS layer against a specific precomputed example."""
        self.d_s_out = 3
        self.hbs = HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=1,
            update_func=None,
        )
        self.set_weights_to_one()
        x_source = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        expected_att_matrix = torch.tensor(
            [
                [0.0, 30.0 / 72.0, 42.0 / 72.0],
                [30.0 / 84.0, 0.0, 54.0 / 84.0],
                [42.0 / 96.0, 54.0 / 96.0, 0.0],
            ],
            dtype=torch.float,
        )
        expected_message = torch.tensor(
            [[3.0, 3.0, 3.0], [7.0, 7.0, 7.0], [11.0, 11.0, 11.0]], dtype=torch.float
        )
        expected_result = torch.mm(expected_att_matrix, expected_message)
        result = self.hbs.forward(x_source, self.neighborhood)
        assert torch.allclose(expected_result, result)
