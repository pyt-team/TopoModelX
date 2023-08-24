"""Test the Dist2Cycle layer."""

import torch

from topomodelx.nn.simplicial.dist2cycle_layer import Dist2CycleLayer


class TestDist2CycleLayer:
    """Test the Dist2Cycle layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        channels = 20
        n_edges = 20
        Linv = torch.randint(0, 2, (n_edges, n_edges)).float()
        adjacency = torch.randint(0, 2, (n_edges, n_edges)).float()

        x_0 = torch.randn(n_edges, n_edges)

        dist2cycle = Dist2CycleLayer(channels)
        output = dist2cycle.forward(x_0, Linv, adjacency)

        assert output.shape[0] == n_edges

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        dist2cycle = Dist2CycleLayer(channels)
        # weights reset and initialized by kaiming_uniform_
        dist2cycle.reset_parameters()
