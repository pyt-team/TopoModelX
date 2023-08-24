"""Test the SCACMPS layer."""

import torch

from topomodelx.base.conv import Conv
from topomodelx.nn.simplicial.sca_cmps_layer import SCACMPSLayer


class TestSCACMPSLayer:
    """Test the SCACMPS layer."""

    def test_sca_cmps_forward(self):
        """Test the forward pass of the SCA layer using CMPS."""
        channels_list = [3, 5, 6, 8]
        n_chains_list = [10, 20, 15, 5]
        down_lap_list = []
        incidence_t_list = []
        for i in range(1, len(n_chains_list)):
            lap_down = torch.randint(0, 2, (n_chains_list[i], n_chains_list[i])).float()
            incidence_transpose = torch.randint(
                0, 2, (n_chains_list[i], n_chains_list[i - 1])
            ).float()
            down_lap_list.append(lap_down)
            incidence_t_list.append(incidence_transpose)

        x_list = []
        for chan, n in zip(channels_list, n_chains_list):
            x = torch.randn(n, chan)
            x_list.append(x)

        sca_cmps = SCACMPSLayer(
            channels_list=channels_list,
            complex_dim=len(n_chains_list),
        )
        output = sca_cmps.forward(x_list, down_lap_list, incidence_t_list)

        for x, n, chan in zip(output, n_chains_list, channels_list):
            assert x.shape == (n, chan)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = [2, 2, 2, 2]
        dim = 4

        sca = SCACMPSLayer(channels, dim)

        initial_params = []
        for module in sca.modules():
            if isinstance(module, torch.nn.ModuleList):
                for sub in module:
                    if isinstance(sub, Conv):
                        initial_params.append(list(sub.parameters()))
                        with torch.no_grad():
                            for param in sub.parameters():
                                param.add_(1.0)

        sca.reset_parameters()
        reset_params = []
        for module in sca.modules():
            if isinstance(module, torch.nn.ModuleList):
                for sub in module:
                    if isinstance(sub, Conv):
                        reset_params.append(list(sub.parameters()))

        count = 0
        for module, reset_param, initial_param in zip(
            sca.modules(), reset_params, initial_params
        ):
            if isinstance(module, torch.nn.ModuleList):
                for sub, r_param, i_param in zip(module, reset_param, initial_param):
                    if isinstance(sub, Conv):
                        torch.testing.assert_close(i_param, r_param)
                        count += 1

        assert count > 0  # Ensuring if-statements were not just failed
