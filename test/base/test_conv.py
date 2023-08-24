"""Test the convolutional layers in the base module."""
import pytest
import torch

from topomodelx.base.conv import Conv


class TestConv:
    """Test the Conv class."""

    def setup_method(self):
        """Set up the test."""
        self.in_channels = 3
        self.out_channels = 5
        self.conv = Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            aggr_norm=True,
            update_func="sigmoid",
            initialization="xavier_uniform",
        )
        self.conv_with_att = Conv(
            in_channels=self.in_channels,
            out_channels=self.out_channels,
            aggr_norm=True,
            update_func="relu",
            initialization="xavier_normal",
            att=True,
        )

        self.conv_without_weight = Conv(
            in_channels=self.in_channels,
            out_channels=self.in_channels,
            aggr_norm=True,
            update_func="relu",
            initialization="xavier_normal",
            with_linear_transform=False,
        )

        self.n_source_cells = 10
        self.n_target_cells = 3
        self.neighborhood = (
            torch.randint(0, 2, (self.n_source_cells, self.n_source_cells))
            .float()
            .to_sparse()
        )
        self.neighborhood_r_to_s = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 0, 1, 2], [0, 1, 1, 2, 9]]),
            values=torch.tensor([1, 2, 3, 4, 5]),
            size=(3, 10),
        )

    def test_update(self):
        """Test the update function."""
        inputs = torch.randn(10, self.out_channels)
        updated = self.conv.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, self.out_channels)

    def test_weight(self):
        """Test learnable weights."""
        inputs = torch.randn(10, self.in_channels)
        updated = self.conv_without_weight.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, self.in_channels)

    def test_validation(self):
        """Test validation of a learnable weight."""
        with pytest.raises(Exception) as exc_info:
            Conv(
                in_channels=self.in_channels,
                out_channels=self.in_channels + 1,
                aggr_norm=True,
                update_func="relu",
                initialization="xavier_normal",
                with_linear_transform=False,
            )
        assert exc_info.type is ValueError
        assert (
            str(exc_info.value)
            == "With `linear_trainsform=False`, in_channels has to be equal to out_channels"
        )

    def test_forward(self):
        """Test the forward pass of the message passing convolution layer."""
        x_source = torch.tensor(
            [
                [1, 2, 2],
                [2, 3, 4],
                [3, 3, 6],
                [4, 4, 5],
                [5, 4, 5],
                [6, 9, 3],
                [7, 3, 4],
                [8, 7, 9],
                [9, 7, 8],
                [10, -1, 2],
            ]
        ).float()
        x_target = torch.tensor([[1, 2, 2], [2, 3, 4], [3, 3, 6]]).float()

        # Without attention
        result = self.conv.forward(x_source, self.neighborhood)
        assert result.shape == (self.n_source_cells, self.out_channels)

        # With attention between cells of the same rank
        result = self.conv_with_att.forward(x_source, self.neighborhood)
        assert result.shape == (self.n_source_cells, self.out_channels)

        # With attention between cells of different ranks
        result = self.conv_with_att.forward(
            x_source, self.neighborhood_r_to_s, x_target
        )
        assert result.shape == (self.n_target_cells, self.out_channels)
