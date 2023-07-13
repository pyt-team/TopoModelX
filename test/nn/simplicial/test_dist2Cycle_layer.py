"""Test the HSN layer."""

import torch

# from topomodelx.nn.simplicial.hsn_layer import HSNLayer
from topomodelx.nn.simplicial.dist2Cycle_layer import Dist2CycleLayer


class TestDist2CycleLayer:
    """Test the HSN layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        channels = 20
        n_nodes = 10
        n_edges = 20
        Linv = torch.randint(0, 2, (n_edges, n_edges)).float()
        adjacency = torch.randint(0, 2, (n_edges, n_edges)).float()

        x_0 = torch.randn(n_edges, n_edges)

        dist2cycle = Dist2CycleLayer(channels)
        print(x_0.shape)
        print(Linv.shape)
        print(adjacency.shape)
        output = dist2cycle.forward(x_0, Linv, adjacency)

        assert output.shape[0] == n_edges

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        dist2cycle = Dist2CycleLayer(channels)
        dist2cycle.reset_parameters()  # weights reset and initilised by kaiming_uniform_

        # for module in dist2cycle.modules():
        #     if isinstance(module, torch.nn.Linear):
        #         torch.testing.assert_allclose(
        #             module.weight, torch.zeros_like(module.weight)
        #         )
        #         torch.testing.assert_allclose(
        #             module.bias, torch.zeros_like(module.bias)
        #         )
