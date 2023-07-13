"""Test the HOAN layer for mesh classification."""

import torch

from topomodelx.base.conv import Conv
from topomodelx.nn.combinatorial.hoan_mc_layer import HOANMCLayer


class TestSCALayer:
    """Test the HSN layer."""

    def test_cmps_forward(self):
        """Test the forward pass of the SCA layer using CMPS."""
        channels_list = [3, 5, 7]
        n_chains_list = [10, 20, 15]

        up_adjacency_0 = (
            torch.randint(-1, 2, (n_chains_list[0], n_chains_list[0]))
            .float()
            .to_sparse()
        )
        incidence_1 = (
            torch.randint(0, 2, (n_chains_list[0], n_chains_list[1]))
            .float()
            .to_sparse()
        )
        up_adjacency_1 = (
            torch.randint(-1, 2, (n_chains_list[1], n_chains_list[1]))
            .float()
            .to_sparse()
        )
        incidence_2 = (
            torch.randint(0, 2, (n_chains_list[1], n_chains_list[2]))
            .float()
            .to_sparse()
        )
        down_adjacency_2 = (
            torch.randint(-1, 2, (n_chains_list[2], n_chains_list[2]))
            .float()
            .to_sparse()
        )

        x_0 = torch.randn(n_chains_list[0], channels_list[0])
        x_1 = torch.randn(n_chains_list[1], channels_list[1])
        x_2 = torch.randn(n_chains_list[2], channels_list[2])

        hoan = HOANMCLayer(
            channels=channels_list,
        )
        x_0f, x_1f, x_2f = hoan.forward(
            x_0=x_0,
            x_1=x_1,
            x_2=x_2,
            up_adjacency_0=up_adjacency_0,
            incidence_1=incidence_1,
            up_adjacency_1=up_adjacency_1,
            incidence_2=incidence_2,
            down_adjacency_2=down_adjacency_2,
        )

        assert x_0f.shape == (n_chains_list[0], channels_list[0])
        assert x_1f.shape == (n_chains_list[1], channels_list[1])
        assert x_2f.shape == (n_chains_list[2], channels_list[2])

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = [2, 2, 2]

        hoan = HOANMCLayer(channels)

        initial_params = []
        for module in hoan.modules():
            if isinstance(module, Conv):
                initial_params.append(list(module.parameters()))
                with torch.no_grad():
                    for param in module.parameters():
                        param.add_(1.0)

        hoan.reset_parameters()
        reset_params = []
        for module in hoan.modules():
            if isinstance(module, Conv):
                reset_params.append(list(module.parameters()))

        count = 0
        for module, reset_param, initial_param in zip(
            hoan.modules(), reset_params, initial_params
        ):
            if isinstance(module, Conv):
                torch.testing.assert_close(initial_param, reset_param)
                count += 1

        assert count > 0  # Ensuring if-statements were not just failed
