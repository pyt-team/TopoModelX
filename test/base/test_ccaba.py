"""Test the convolutional layers in the base module."""

import torch

from topomodelx.base.ccaba import CCABA


class TestCCABA:
    """Test the CCABA class."""

    def setup_method(self):
        """Set up the test."""
        self.d_s_in, self.d_s_out = 2, 3

        self.ccaba = CCABA(
            d_s_in=self.d_s_in,
            d_s_out=self.d_s_out,
            negative_slope=0.2,
            aggr_norm=True,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )

        self.n_source_cells = 10

        self.neighborhood = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 1, 2, 9],[3, 7, 9, 2, 5]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(10,10),
        )

    def test_forward(self):
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
            ]
        ).float()

        result = self.ccaba.forward(x_source, self.neighborhood)

        assert result.shape == (self.n_source_cells, self.d_s_out)

    """
        def test_attention(self):
            s_message = torch.tensor(
                [
                    [1, 2, 2, 1],
                    [2, 3, 3, 2],
                    [3, 3, 3, 3],
                    [4, 4, 4, 4],
                    [5, 4, 4, 5],
                    [6, 9, 9, 6],
                    [7, 3, 3, 7],
                    [8, 7, 7, 8],
                    [9, 7, 7, 9],
                    [10, -1, -1, 10]
                ]
            ).float()

            t_message = torch.tensor([[1, 2, 2], [2, 3, 4], [3, 3, 6]]).float()

            s_t_attention, t_s_attention = self.ccat.attention(s_message, t_message)

            assert s_t_attention.shape == (self.n_source_cells, self.n_target_cells)
            assert t_s_attention.shape == (self.n_target_cells, self.n_source_cells)"""
