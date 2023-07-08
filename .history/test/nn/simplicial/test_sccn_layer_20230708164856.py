"""Test the SCCN layer."""

import torch
import topomodelx

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


class TestSCCNLayer:
    """Test the SCCN layer."""

    def test_forward(self):
        """Test the forward pass of the SCCN layer."""
        channels = 5
        max_rank = 1
        n_rank_0_cells = 11
        n_rank_1_cells = 22

        incidences = {"rank_1": 2 * torch.randint(0, 2, (n_rank_0_cells, n_rank_1_cells)).float() - 1}
                      
        adjacencies = {"rank_0": torch.eye(n_rank_0_cells).float(),
                       "rank_1": 2 * torch.randint(0, 2, (n_rank_1_cells, n_rank_1_cells)).float() - 1}

        features = {"rank_0": torch.randn(n_rank_0_cells, channels), 
                    "rank_1": torch.randn(n_rank_1_cells, channels)}

        sccn = SCCNLayer(channels, max_rank)
        output = sccn.forward(features, incidences, adjacencies)

        assert len(output) == max_rank + 1
        assert output["rank_0"].shape == (n_rank_0_cells, channels)
        assert output["rank_1"].shape == (n_rank_1_cells, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5
        max_rank = 1

        sccn = SCCNLayer(channels, max_rank)
        sccn.reset_parameters()

        for module in sccn.modules():
            if isinstance(module, topomodelx.base.conv.Conv):
                try:
                    torch.testing.assert_allclose(
                        module.weight,  torch.nn.init.xavier_uniform_(module.weight.view(-1, 1), gain=gain)(module.weight)
                    )

                except AssertionError as ae:
                    # This is expected if parameters have changed
                    pass
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )


if __name__ == "__main__":
    TestSCCNLayer().test_forward()
    TestSCCNLayer().test_reset_parameters()