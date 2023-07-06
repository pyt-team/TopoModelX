"""Unit tests for the CANLayer class."""

import pytest
import torch

from topomodelx.nn.cell.canv2_layer import CANLayer
import itertools

class TestCANLayer:
    """Unit tests for the CANLayer class."""

    # def test_forward(self):
    #     """Test the forward method of CANLayer."""
    #     in_channels = 7
    #     out_channels = 64
    #     dropout = 0.5
    #     heads = 3
    #     concat = True
    #     skip_connection = True

    #     n_cells = 21

    #     x_1 = torch.randn(n_cells, in_channels)

    #     lower_neighborhood = torch.randn(n_cells, n_cells)
    #     upper_neighborhood = torch.randn(n_cells, n_cells)

    #     lower_neighborhood = lower_neighborhood.to_sparse().float()
    #     upper_neighborhood = upper_neighborhood.to_sparse().float()

    #     can_layer = CANLayer(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         dropout=dropout,
    #         heads=heads,
    #         concat=concat,
    #         skip_connection=skip_connection,
    #     )
    #     x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # test forward with different number of heads predifined arguments
    #     heads = 3
    #     can_layer = CANLayer(
    #         in_channels=in_channels, out_channels=out_channels, heads=heads
    #     )
    #     x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # test forward without skip connection predifined arguments
    #     skip_connection = False
    #     heads = 1
    #     can_layer = CANLayer(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         skip_connection=skip_connection,
    #     )
    #     x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # test forward without concat and predifined arguments
    #     concat = False
    #     heads = 1
    #     skip_connection = True
    #     can_layer = CANLayer(
    #         in_channels=in_channels, out_channels=out_channels, concat=concat
    #     )
    #     x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # test forward with self loops
    #     heads = 1
    #     concat = True
    #     skip_connection = True
    #     can_layer = CANLayer(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         concat=concat,
    #         skip_connection=skip_connection,
    #         add_self_loops=True,
    #     )
    #     x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # Test if there are no non-zero values in the neighborhood
    #     heads = 1
    #     concat = True
    #     skip_connection = True
    #     can_layer = CANLayer(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         concat=concat,
    #         skip_connection=skip_connection,
    #     )
    #     x_out = can_layer.forward(
    #         x_1,
    #         torch.zeros_like(lower_neighborhood),
    #         torch.zeros_like(upper_neighborhood),
    #     )
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    #     # Now without concat
    #     concat = False
    #     heads = 1
    #     skip_connection = True
    #     can_layer = CANLayer(
    #         in_channels=in_channels,
    #         out_channels=out_channels,
    #         concat=concat,
    #         skip_connection=skip_connection,
    #     )
    #     x_out = can_layer.forward(
    #         x_1,
    #         torch.zeros_like(lower_neighborhood),
    #         torch.zeros_like(upper_neighborhood),
    #     )
    #     if concat:
    #         assert x_out.shape == (n_cells, out_channels * heads)
    #     else:
    #         assert x_out.shape == (n_cells, out_channels)

    def test_reset_parameters(self):
        """Test the reset_parameters method of CANLayer."""
        in_channels = 2
        out_channels = 5

        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
        )
        can_layer.reset_parameters()

        for module in can_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()

    def test_forward(self):
        in_channels = 7
        out_channels = 64
        dropout_values = [0.5, 0.7]
        heads_values = [1, 3]
        concat_values = [True, False]
        skip_connection_values = [True, False]
        self_loop_values = [True, False]

        n_cells = 21
        x_1 = torch.randn(n_cells, in_channels)
        lower_neighborhood = torch.randn(n_cells, n_cells)
        upper_neighborhood = torch.randn(n_cells, n_cells)
        lower_neighborhood = lower_neighborhood.to_sparse().float()
        upper_neighborhood = upper_neighborhood.to_sparse().float()

        all_parameter_combinations = itertools.product(
            dropout_values,
            heads_values,
            concat_values,
            skip_connection_values,
            self_loop_values,
        )

        for parameters in all_parameter_combinations:
            dropout, heads, concat, skip_connection, self_loops = parameters
            can_layer = CANLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                heads=heads,
                concat=concat,
                skip_connection=skip_connection,
                add_self_loops = self_loops

            )
            x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
            if concat:
                assert x_out.shape == (n_cells, out_channels * heads)
            else:
                assert x_out.shape == (n_cells, out_channels)

        # Test if there are no non-zero values in the neighborhood
        heads = 1
        concat = True
        skip_connection = True
        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            concat=concat,
            skip_connection=skip_connection,
        )
        x_out = can_layer.forward(
            x_1,
            torch.zeros_like(lower_neighborhood),
            torch.zeros_like(upper_neighborhood),
        )
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)

        # Now without concat
        concat = False
        heads = 1
        skip_connection = True
        can_layer = CANLayer(
            in_channels=in_channels,
            out_channels=out_channels,
            concat=concat,
            skip_connection=skip_connection,
        )
        x_out = can_layer.forward(
            x_1,
            torch.zeros_like(lower_neighborhood),
            torch.zeros_like(upper_neighborhood),
        )
        if concat:
            assert x_out.shape == (n_cells, out_channels * heads)
        else:
            assert x_out.shape == (n_cells, out_channels)