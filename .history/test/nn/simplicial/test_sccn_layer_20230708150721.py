"""Test the SCCN layer."""

import torch

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


class TestSCCNLayer:
    """Test the SCCN layer."""

    def test_forward(self):
        """Test the forward pass of the SCCN layer."""
        channels = 5
        max_rank = 1
        n_rank_0_cells = 5
        n_rank_1_cells = 10

        incidences = {"rank_0": torch.randint(0, 2, (n_rank_1_cells, n_ran)).float()
                      
        adjacencies = torch.randint(0, 2, (n_nodes, n_nodes)).float()

        features = {"rank_0": torch.randn(n_rank_0_cells, channels), 
                    "rank_1": torch.randn(n_rank_1_cells, channels)}

        sccn = SCCNLayer(channels, max_rank)
        output = sccn.forward(features, incidence_1, adjacency_0)

        assert len(output) == max_rank + 1
        assert output.shape == (n_rank_1_cells, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5
        max_rank = 1

        sccn = SCCNLayer(channels, max_rank)
        sccn.reset_parameters()

        for module in sccn.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )
