"""Test the Higher Order Attention Block for non-squared neighborhoods (HBNS) layer in the base module."""
import math

import torch

from topomodelx.base.hbns import HBNS


class TestHBNS:
    """Test the HBNS class."""
    def set_constant_weights(self):
        """Set the weights to constant values."""
        with torch.no_grad():
            self.hbns.att_weight[:2] = 2.0
            self.hbns.att_weight[2:] = 1.0
        torch.nn.init.constant_(self.hbns.w_s, 1.0)
        torch.nn.init.constant_(self.hbns.w_t, 1.0)

    def setup_method(self):
        """Set up the test."""
        self.d_s_in, self.d_s_out = 2, 2
        self.d_t_in, self.d_t_out = 2, 2

        self.hbns = HBNS(source_in_channels=self.d_s_in, source_out_channels=self.d_s_out,
                         target_in_channels=self.d_t_in, target_out_channels=self.d_t_out, negative_slope=0.2,
                         softmax=False, update_func="sigmoid", initialization="xavier_uniform")

        self.neighborhood_s_to_t = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
        )

    def test_forward_shape(self):
        """Test the shapes of the outputs of the forward pass of the HBNS layer."""
        self.d_s_in, self.d_s_out = 2, 3
        self.d_t_in, self.d_t_out = 3, 4

        self.hbns = HBNS(source_in_channels=self.d_s_in, source_out_channels=self.d_s_out,
                         target_in_channels=self.d_t_in, target_out_channels=self.d_t_out, negative_slope=0.2,
                         update_func="sigmoid", initialization="xavier_uniform")

        self.n_source_cells = 10
        self.n_target_cells = 3

        self.neighborhood_s_to_t = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 2], [0, 1, 1, 2, 9]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(3, 10),
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
                [10, -1]
            ], dtype=torch.float
        )

        x_target = torch.tensor([[1, 2, 2], [2, 3, 4], [3, 3, 6]], dtype=torch.float)

        result = self.hbns.forward(
            x_source, x_target, self.neighborhood_s_to_t
        )

        message_on_source, message_on_target = result

        assert message_on_source.shape == (self.n_source_cells, self.d_s_out)
        assert message_on_target.shape == (self.n_target_cells, self.d_t_out)

    def test_attention_without_softmax(self):
        """Test the calculation of the attention matrices calculation without softmax."""
        self.set_constant_weights()
        # Create the message that will be used for the attention.
        target_message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        source_message = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
        # Calculate the attention matrix.
        neighborhood_s_to_t = self.neighborhood_s_to_t.coalesce()
        neighborhood_t_to_s = self.neighborhood_s_to_t.t().coalesce()

        self.hbns.target_indices, self.hbns.source_indices = neighborhood_s_to_t.indices()

        att_matrix_s_to_t, att_matrix_t_to_s = self.hbns.attention(source_message, target_message)
        att_matrix_s_to_t, att_matrix_t_to_s = att_matrix_s_to_t.to_dense(), att_matrix_t_to_s.to_dense()
        # Create the expected attention matrix. The values have been calculated by hand.
        expected_att_matrix_s_to_t = neighborhood_s_to_t.to_dense() * torch.tensor(
            [
                [33.0 / 82.0, 41.0 / 82.0, 49.0 / 82.0],
                [37.0 / 82.0, 45.0 / 82.0, 53.0 / 82.0],
                [41.0 / 106.0, 49.0 / 106.0, 57.0 / 106.0]
            ],
            dtype=torch.float
        )
        expected_att_matrix_t_to_s = neighborhood_t_to_s.to_dense() * torch.tensor(
            [
                [33.0 / 70.0, 37.0 / 70.0, 41.0 / 70.0],
                [41.0 / 94.0, 45.0 / 94.0, 49.0 / 94.0],
                [49.0 / 106.0, 53.0 / 106.0, 57.0 / 106.0]
            ],
            dtype=torch.float
        )

        # Compare the two attention matrices.
        assert torch.allclose(att_matrix_s_to_t, expected_att_matrix_s_to_t)
        assert torch.allclose(att_matrix_t_to_s, expected_att_matrix_t_to_s)

    def test_attention_with_softmax(self):
        """Test the calculation of the attention matrices calculation with softmax."""
        self.set_constant_weights()
        self.hbns.softmax = True
        # Create the message that will be used for the attention.
        target_message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        source_message = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
        # Calculate the attention matrix.
        neighborhood_s_to_t = self.neighborhood_s_to_t.coalesce()
        neighborhood_t_to_s = self.neighborhood_s_to_t.t().coalesce()

        self.hbns.target_indices, self.hbns.source_indices = neighborhood_s_to_t.indices()

        att_matrix_s_to_t, att_matrix_t_to_s = self.hbns.attention(source_message, target_message)
        att_matrix_s_to_t, att_matrix_t_to_s = att_matrix_s_to_t.to_dense(), att_matrix_t_to_s.to_dense()
        # Create the expected attention matrix. The values have been calculated by hand.
        expected_att_s_to_t_wo_product = torch.exp(torch.tensor(
            [
                [33.0, 41.0, 49.0],
                [37.0, 45.0, 53.0],
                [41.0, 49.0, 57.0]
            ],
            dtype=torch.float
        ))
        row_normalization_denominator = torch.tensor([1.0 / (math.exp(33.0) + math.exp(49.0)),
                                                      1.0 / (math.exp(37.0) + math.exp(45.0)),
                                                      1.0 / (math.exp(49.0) + math.exp(57.0))])
        expected_att_s_to_t_matrix_wo_product = (expected_att_s_to_t_wo_product.T * row_normalization_denominator).T
        expected_att_matrix_s_to_t = neighborhood_s_to_t.to_dense() * expected_att_s_to_t_matrix_wo_product

        expected_att_t_to_s_wo_product = torch.exp(torch.tensor(
            [
                [33.0, 37.0, 41.0],
                [41.0, 45.0, 49.0],
                [49.0, 53.0, 57.0]
            ],
            dtype=torch.float
        ))
        row_normalization_denominator = torch.tensor([1.0 / (math.exp(33.0) + math.exp(37.0)),
                                                      1.0 / (math.exp(45.0) + math.exp(49.0)),
                                                      1.0 / (math.exp(49.0) + math.exp(57.0))])
        expected_att_t_to_s_matrix_wo_product = (expected_att_t_to_s_wo_product.T * row_normalization_denominator).T
        expected_att_matrix_t_to_s = neighborhood_t_to_s.to_dense() * expected_att_t_to_s_matrix_wo_product

        # Compare the two attention matrices.
        assert torch.allclose(att_matrix_s_to_t, expected_att_matrix_s_to_t)
        assert torch.allclose(att_matrix_t_to_s, expected_att_matrix_t_to_s)

    def test_forward_values(self):
        """Test the values of the outputs of the forward pass of the HBNS layer against a specific precomputed example.
        """
        self.d_s_in, self.d_s_out = 2, 3
        self.d_t_in, self.d_t_out = 2, 3

        self.hbns = HBNS(source_in_channels=self.d_s_in, source_out_channels=self.d_s_out,
                         target_in_channels=self.d_t_in, target_out_channels=self.d_t_out, negative_slope=0.2,
                         softmax=False, update_func=None, initialization="xavier_uniform")

        with torch.no_grad():
            self.hbns.att_weight[:3] = 1.0
            self.hbns.att_weight[3:] = 2.0
        torch.nn.init.constant_(self.hbns.w_s, 1.0)
        torch.nn.init.constant_(self.hbns.w_t, 1.0)

        self.neighborhood_s_to_t = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
        )
        x_source = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        x_target = torch.tensor([[5, 6], [7, 8], [9, 10]], dtype=torch.float)

        expected_s_message = torch.tensor(
            [
                [3, 3, 3],
                [7, 7, 7],
                [11, 11, 11]
            ], dtype=torch.float
        )
        expected_t_message = torch.tensor(
            [
                [11, 11, 11],
                [15, 15, 15],
                [19, 19, 19]
            ], dtype=torch.float
        )
        expected_att_matrix_s_to_t = torch.tensor(
            [
                [75.0 / 174.0, 0.0, 99.0 / 174.0],
                [99.0 / 210.0, 111.0 / 210.0, 0.0],
                [0.0, 135.0 / 282.0, 147.0 / 282.0]
            ], dtype=torch.float
        )
        expected_att_matrix_t_to_s = torch.tensor(
            [
                [75.0 / 174.0, 99.0 / 174.0, 0.0],
                [0.0, 111.0 / 246.0, 135.0 / 246.0],
                [99.0 / 246.0, 0.0, 147.0 / 246.0]
            ], dtype=torch.float
        )
        expected_message_on_target = torch.mm(expected_att_matrix_s_to_t, expected_s_message)
        expected_message_on_source = torch.mm(expected_att_matrix_t_to_s, expected_t_message)

        message_on_source, message_on_target = self.hbns.forward(x_source, x_target, self.neighborhood_s_to_t)

        assert torch.allclose(expected_message_on_source, message_on_source)
        assert torch.allclose(expected_message_on_target, message_on_target)
