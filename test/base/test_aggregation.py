"""Test the merge layer of the topomodelx base module."""


import torch

from topomodelx.base.aggregation import Aggregation


class TestAggregation:
    """Test the Aggregation class."""

    def test_update(self):
        """Test the update of messages."""
        n_target_cells = 4
        out_channels = 8
        inputs = torch.randn(n_target_cells, out_channels)
        merge_layer = Aggregation(update_func="sigmoid")
        updated = merge_layer.update(inputs)
        assert updated.shape == (n_target_cells, out_channels)

    def test_forward(self):
        """Test the forward pass of the merge layer."""
        n_messages_to_merge = 2
        n_cells_on_merging_rank = 6
        out_channels = 8
        inputs = [
            torch.randn(n_cells_on_merging_rank, out_channels)
            for _ in range(n_messages_to_merge)
        ]
        merge_layer = Aggregation()
        merged = merge_layer.forward(inputs)
        assert merged.shape == (n_cells_on_merging_rank, out_channels)
