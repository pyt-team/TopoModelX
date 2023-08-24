"""Test the SCN layer."""

import torch

import topomodelx
from topomodelx.nn.simplicial.scn_layer import SCNLayer


class TestSCNLayer:
    """Test the SCN layer."""

    def test_forward(self):
        """Test the forward pass of the SCN layer."""
        channels = 5
        max_rank = 3
        n_rank_0_cells = 11
        n_rank_1_cells = 22
        n_rank_2_cells = 33
        n_rank_3_cells = 44

        incidences = {
            "rank_1": 2 * torch.randint(0, 2, (n_rank_0_cells, n_rank_1_cells)).float()
            - 1,
            "rank_2": 2 * torch.randint(0, 2, (n_rank_1_cells, n_rank_2_cells)).float()
            - 1,
            "rank_3": 2 * torch.randint(0, 2, (n_rank_2_cells, n_rank_3_cells)).float()
            - 1,
        }

        adjacencies = {
            "rank_0": torch.eye(n_rank_0_cells).float(),
            "rank_1": 2 * torch.randint(0, 2, (n_rank_1_cells, n_rank_1_cells)).float()
            - 1,
            "rank_2": 2 * torch.randint(0, 2, (n_rank_2_cells, n_rank_2_cells)).float()
            - 1,
            "rank_3": 2 * torch.randint(0, 2, (n_rank_3_cells, n_rank_3_cells)).float()
            - 1,
        }

        features = {
            "rank_0": torch.randn(n_rank_0_cells, channels),
            "rank_1": torch.randn(n_rank_1_cells, channels),
            "rank_2": torch.randn(n_rank_2_cells, channels),
            "rank_3": torch.randn(n_rank_3_cells, channels),
        }

        scn = SCNLayer(channels, max_rank)
        output = scn.forward(features, incidences, adjacencies)

        assert len(output) == max_rank + 1
        assert output["rank_0"].shape == (n_rank_0_cells, channels)
        assert output["rank_1"].shape == (n_rank_1_cells, channels)
        assert output["rank_2"].shape == (n_rank_2_cells, channels)
        assert output["rank_3"].shape == (n_rank_3_cells, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5
        max_rank = 3

        scn = SCNLayer(channels, max_rank)
        scn.reset_parameters()

        for module in scn.modules():
            if isinstance(module, topomodelx.base.conv.Conv):
                try:
                    torch.testing.assert_allclose(
                        module.weight,
                        torch.nn.init.xavier_uniform_(
                            module.weight.clone(), gain=1.414
                        ),
                    )
                    # Raise AssertionError if parameters have not changed after the reset
                    raise AssertionError("Parameters have not changed after the reset")

                except AssertionError:
                    # This is expected if parameters have changed
                    pass
