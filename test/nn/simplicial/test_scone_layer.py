"""Test the SCoNe Layer."""

import torch

from topomodelx.base.conv import Conv
from topomodelx.nn.simplicial.scone_layer import SCoNeLayer


class TestSCoNeLayer:
    """Test the SCoNe Layer."""

    def test_forward(self):
        """Test the forward pass of the HSN layer."""
        channels = 5
        n_edges = 20
        up_lap1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        down_lap1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        iden = torch.eye(n_edges)

        x_1 = torch.randn(n_edges, channels)

        scone = SCoNeLayer(channels)
        output = scone.forward(x_1, up_lap1, down_lap1, iden)

        assert output.shape == (n_edges, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        channels = 5

        scone = SCoNeLayer(channels)

        initial_params = []
        for module in scone.modules():
            if isinstance(module, Conv):
                initial_params.append(list(module.parameters()))
                with torch.no_grad():
                    for param in module.parameters():
                        param.add_(1.0)

        scone.reset_parameters()
        reset_params = []
        for module in scone.modules():
            if isinstance(module, Conv):
                reset_params.append(list(module.parameters()))

        count = 0
        for module, reset_param, initial_param in zip(
            scone.modules(), reset_params, initial_params
        ):
            if isinstance(module, Conv):
                torch.testing.assert_close(initial_param, reset_param)
                count += 1

        assert count > 0  # Ensuring if-statements were not just failed
