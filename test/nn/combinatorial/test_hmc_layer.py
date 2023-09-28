"""Unit tests for the HMCLayer class."""
import math

import pytest
import torch

from topomodelx.nn.combinatorial.hmc_layer import HBNS, HBS, HMCLayer


class TestHMCLayer:
    """Unit tests for the HMCLayer class."""

    def test_forward(self):
        """Test the forward method of HMCLayer."""
        in_channels = [1, 2, 3]
        intermediate_channels = [3, 4, 5]
        out_channels = [7, 8, 9]
        hmc_layer = HMCLayer(in_channels, intermediate_channels, out_channels, 0.1)
        adjacency_0 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        adjacency_1 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        coadjacency_2 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1]),
            size=(1, 1),
            dtype=torch.float,
        )
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        incidence_2 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2], [0, 0, 0]]),
            values=torch.tensor([1, 1, 1]),
            size=(3, 1),
            dtype=torch.float,
        )
        x_0 = torch.randn(3, in_channels[0])
        x_1 = torch.randn(3, in_channels[1])
        x_2 = torch.randn(1, in_channels[2])

        x_0_out, x_1_out, x_2_out = hmc_layer.forward(
            x_0,
            x_1,
            x_2,
            adjacency_0,
            adjacency_1,
            coadjacency_2,
            incidence_1,
            incidence_2,
        )
        assert x_0_out.shape == (3, out_channels[0])
        assert x_1_out.shape == (3, out_channels[1])
        assert x_2_out.shape == (1, out_channels[2])


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

        self.hbns = HBNS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            target_in_channels=self.d_t_in,
            target_out_channels=self.d_t_out,
            negative_slope=0.2,
            softmax=False,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )

        self.neighborhood_s_to_t = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
        )

    def test_initialization(self):
        """Test if the initialization of the parameter weights works."""
        HBNS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            target_in_channels=self.d_t_in,
            target_out_channels=self.d_t_out,
            negative_slope=0.2,
            softmax=False,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )
        HBNS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            target_in_channels=self.d_t_in,
            target_out_channels=self.d_t_out,
            negative_slope=0.2,
            softmax=False,
            update_func="sigmoid",
            initialization="xavier_normal",
        )
        with pytest.raises((RuntimeError, AssertionError)):
            HBNS(
                source_in_channels=self.d_s_in,
                source_out_channels=self.d_s_out,
                target_in_channels=self.d_t_in,
                target_out_channels=self.d_t_out,
                negative_slope=0.2,
                softmax=False,
                update_func="sigmoid",
                initialization="non_existing",
            )

    def test_forward_shape(self):
        """Test the shapes of the outputs of the forward pass."""
        self.d_s_in, self.d_s_out = 2, 3
        self.d_t_in, self.d_t_out = 3, 4

        self.hbns = HBNS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            target_in_channels=self.d_t_in,
            target_out_channels=self.d_t_out,
            negative_slope=0.2,
            update_func="relu",
            initialization="xavier_normal",
        )

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
                [10, -1],
            ],
            dtype=torch.float,
        )

        x_target = torch.tensor([[1, 2, 2], [2, 3, 4], [3, 3, 6]], dtype=torch.float)

        result = self.hbns.forward(x_source, x_target, self.neighborhood_s_to_t)

        message_on_source, message_on_target = result

        assert message_on_source.shape == (self.n_source_cells, self.d_s_out)
        assert message_on_target.shape == (self.n_target_cells, self.d_t_out)

    def test_attention_without_softmax(self):
        """Test the calculation of the attention matrices without softmax."""
        self.set_constant_weights()
        # Create the message that will be used for the attention.
        target_message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        source_message = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
        # Calculate the attention matrix.
        neighborhood_s_to_t = self.neighborhood_s_to_t.coalesce()
        neighborhood_t_to_s = self.neighborhood_s_to_t.t().coalesce()

        (
            self.hbns.target_indices,
            self.hbns.source_indices,
        ) = neighborhood_s_to_t.indices()

        att_matrix_s_to_t, att_matrix_t_to_s = self.hbns.attention(
            source_message, target_message
        )
        att_matrix_s_to_t, att_matrix_t_to_s = (
            att_matrix_s_to_t.to_dense(),
            att_matrix_t_to_s.to_dense(),
        )
        # Expected attention matrix. The values have been calculated by hand.
        expected_att_matrix_s_to_t = neighborhood_s_to_t.to_dense() * torch.tensor(
            [
                [33.0 / 82.0, 41.0 / 82.0, 49.0 / 82.0],
                [37.0 / 82.0, 45.0 / 82.0, 53.0 / 82.0],
                [41.0 / 106.0, 49.0 / 106.0, 57.0 / 106.0],
            ],
            dtype=torch.float,
        )
        expected_att_matrix_t_to_s = neighborhood_t_to_s.to_dense() * torch.tensor(
            [
                [33.0 / 70.0, 37.0 / 70.0, 41.0 / 70.0],
                [41.0 / 94.0, 45.0 / 94.0, 49.0 / 94.0],
                [49.0 / 106.0, 53.0 / 106.0, 57.0 / 106.0],
            ],
            dtype=torch.float,
        )

        # Compare the two attention matrices.
        assert torch.allclose(att_matrix_s_to_t, expected_att_matrix_s_to_t)
        assert torch.allclose(att_matrix_t_to_s, expected_att_matrix_t_to_s)

    def test_attention_with_softmax(self):
        """Test the calculation of the attention matrices with softmax."""
        self.set_constant_weights()
        self.hbns.softmax = True
        # Create the message that will be used for the attention.
        target_message = torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.float)
        source_message = torch.tensor([[7, 8], [9, 10], [11, 12]], dtype=torch.float)
        # Calculate the attention matrix.
        neighborhood_s_to_t = self.neighborhood_s_to_t.coalesce()
        neighborhood_t_to_s = self.neighborhood_s_to_t.t().coalesce()

        (
            self.hbns.target_indices,
            self.hbns.source_indices,
        ) = neighborhood_s_to_t.indices()

        att_matrix_s_to_t, att_matrix_t_to_s = self.hbns.attention(
            source_message, target_message
        )
        att_matrix_s_to_t, att_matrix_t_to_s = (
            att_matrix_s_to_t.to_dense(),
            att_matrix_t_to_s.to_dense(),
        )
        # Expected attention matrix. The values have been calculated by hand.
        expected_att_s_to_t_wo_product = torch.exp(
            torch.tensor(
                [[33.0, 41.0, 49.0], [37.0, 45.0, 53.0], [41.0, 49.0, 57.0]],
                dtype=torch.float,
            )
        )
        row_normalization_denominator = torch.tensor(
            [
                1.0 / (math.exp(33.0) + math.exp(49.0)),
                1.0 / (math.exp(37.0) + math.exp(45.0)),
                1.0 / (math.exp(49.0) + math.exp(57.0)),
            ]
        )
        expected_att_s_to_t_matrix_wo_product = (
            expected_att_s_to_t_wo_product.T * row_normalization_denominator
        ).T
        expected_att_matrix_s_to_t = (
            neighborhood_s_to_t.to_dense() * expected_att_s_to_t_matrix_wo_product
        )

        expected_att_t_to_s_wo_product = torch.exp(
            torch.tensor(
                [[33.0, 37.0, 41.0], [41.0, 45.0, 49.0], [49.0, 53.0, 57.0]],
                dtype=torch.float,
            )
        )
        row_normalization_denominator = torch.tensor(
            [
                1.0 / (math.exp(33.0) + math.exp(37.0)),
                1.0 / (math.exp(45.0) + math.exp(49.0)),
                1.0 / (math.exp(49.0) + math.exp(57.0)),
            ]
        )
        expected_att_t_to_s_matrix_wo_product = (
            expected_att_t_to_s_wo_product.T * row_normalization_denominator
        ).T
        expected_att_matrix_t_to_s = (
            neighborhood_t_to_s.to_dense() * expected_att_t_to_s_matrix_wo_product
        )

        # Compare the two attention matrices.
        assert torch.allclose(att_matrix_s_to_t, expected_att_matrix_s_to_t)
        assert torch.allclose(att_matrix_t_to_s, expected_att_matrix_t_to_s)

    def test_forward_values(self):
        """Test the outputs of the forward pass of the HBNS layer."""
        self.d_s_in, self.d_s_out = 2, 3
        self.d_t_in, self.d_t_out = 2, 3

        self.hbns = HBNS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            target_in_channels=self.d_t_in,
            target_out_channels=self.d_t_out,
            negative_slope=0.2,
            softmax=False,
            update_func=None,
            initialization="xavier_uniform",
        )

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
            [[3, 3, 3], [7, 7, 7], [11, 11, 11]], dtype=torch.float
        )
        expected_t_message = torch.tensor(
            [[11, 11, 11], [15, 15, 15], [19, 19, 19]], dtype=torch.float
        )
        expected_att_matrix_s_to_t = torch.tensor(
            [
                [75.0 / 174.0, 0.0, 99.0 / 174.0],
                [99.0 / 210.0, 111.0 / 210.0, 0.0],
                [0.0, 135.0 / 282.0, 147.0 / 282.0],
            ],
            dtype=torch.float,
        )
        expected_att_matrix_t_to_s = torch.tensor(
            [
                [75.0 / 174.0, 99.0 / 174.0, 0.0],
                [0.0, 111.0 / 246.0, 135.0 / 246.0],
                [99.0 / 246.0, 0.0, 147.0 / 246.0],
            ],
            dtype=torch.float,
        )
        expected_message_on_target = torch.mm(
            expected_att_matrix_s_to_t, expected_s_message
        )
        expected_message_on_source = torch.mm(
            expected_att_matrix_t_to_s, expected_t_message
        )

        message_on_source, message_on_target = self.hbns.forward(
            x_source, x_target, self.neighborhood_s_to_t
        )

        assert torch.allclose(expected_message_on_source, message_on_source)
        assert torch.allclose(expected_message_on_target, message_on_target)


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
        """Test the shapes of the outputs of the forward pass."""
        self.d_s_out = 3
        self.hbs = HBS(
            source_in_channels=self.d_s_in,
            source_out_channels=self.d_s_out,
            negative_slope=0.2,
            softmax=False,
            m_hop=2,
            update_func="relu",
            initialization="xavier_normal",
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
        with pytest.raises((RuntimeError, AssertionError)):
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
        # Expected attention matrix. The values have been calculated by hand.
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
        # Expected attention matrix. The values have been calculated by hand.
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
        """Test the forward pass of the HBS layer against an example."""
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
