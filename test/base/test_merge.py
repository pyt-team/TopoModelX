"""Test the merge layer of the topomodelx base module."""


import torch

from topomodelx.base.merge import _Merge


class TestMerge:
    """Test the Merge class."""

    def test_aggregate(self):
        """Test the aggregation of messages."""
        n_neighborhoods = 2
        n_skeleton_out = 4
        out_channels = 8
        inputs = [
            torch.randn(n_skeleton_out, out_channels) for _ in range(n_neighborhoods)
        ]
        merge_layer = _Merge()
        aggregated = merge_layer.aggregate(inputs)
        assert aggregated.shape == (n_skeleton_out, out_channels)

    def test_update(self):
        """Test the update of messages."""
        n_skeleton_out = 4
        out_channels = 8
        inputs = torch.randn(n_skeleton_out, out_channels)
        merge_layer = _Merge(update_on_merge="sigmoid")
        updated = merge_layer.update(inputs)
        assert updated.shape == (n_skeleton_out, out_channels)

    def test_forward(self):
        """Test the forward pass of the merge layer."""
        n_messages_to_merge = 2
        n_cells_on_merging_rank = 6
        out_channels = 8
        inputs = [
            torch.randn(n_cells_on_merging_rank, out_channels)
            for _ in range(n_messages_to_merge)
        ]
        merge_layer = _Merge()
        merged = merge_layer.forward(inputs)
        assert merged.shape == (n_cells_on_merging_rank, out_channels)
