"""Unit tests for the CANLayer class, the PoolLayer class and the Attentional Lift class."""

import itertools

import torch

from topomodelx.nn.cell.can_layer_final import CANLayer, MultiHeadLiftLayer, PoolLayer


class TestCANLayer:
    """Unit tests for the CANLayer class."""

    def test_forward(self):
        """Test the forward method of CANLayer."""
        in_channels = 7
        out_channels = 64
        dropout_values = [0.5, 0.7]
        heads_values = [1, 3]
        concat_values = [True, False]
        skip_connection_values = [True, False]
        self_loop_values = [True, False]
        version = ["v1", "v2"]
        shared_weights = [True, False]

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
            version,
            shared_weights,
        )

        for parameters in all_parameter_combinations:
            (
                dropout,
                heads,
                concat,
                skip_connection,
                self_loops,
                version,
                shared_weights,
            ) = parameters
            # Skip the case where the version is v1 and the weights are shared
            if version == "v1" and shared_weights is True:
                continue
            can_layer = CANLayer(
                in_channels=in_channels,
                out_channels=out_channels,
                dropout=dropout,
                heads=heads,
                concat=concat,
                skip_connection=skip_connection,
                add_self_loops=self_loops,
                version=version,
                share_weights=shared_weights,
            )
            x_out = can_layer.forward(x_1, lower_neighborhood, upper_neighborhood)
            if concat:
                assert x_out.shape == (n_cells, out_channels * heads)
            else:
                assert x_out.shape == (n_cells, out_channels)

        # Test if there are no non-zero values in the neighborhood
        heads = 1
        concat = [True, False]
        skip_connection = True

        for concat in concat:
            for version in ["v1", "v2"]:
                can_layer = CANLayer(
                    in_channels=in_channels,
                    out_channels=out_channels,
                    concat=concat,
                    skip_connection=skip_connection,
                    version=version,
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


class TestPoolLayer:
    """Unit tests for the PoolLayer class."""

    def test_forward(self):
        """Test the forward method of PoolLayer."""
        k_pool = 0.75
        in_channels_0 = 96
        signal_pool_activation = torch.nn.ReLU()

        # Input
        x_0 = torch.randn(38, in_channels_0)
        lower_neighborhood = torch.randn(38, 38)
        upper_neighborhood = torch.randn(38, 38)

        # Instantiate the PoolLayer
        pool_layer = PoolLayer(
            in_channels_0=in_channels_0,
            k_pool=k_pool,
            signal_pool_activation=signal_pool_activation,
            readout=True,
        )
        out, lower_neighborhood, upper_neighborhood = pool_layer.forward(
            x_0, lower_neighborhood, upper_neighborhood
        )
        assert out.shape == (int(k_pool * x_0.size(0)), in_channels_0)
        assert lower_neighborhood.shape == (
            int(k_pool * x_0.size(0)),
            int(k_pool * x_0.size(0)),
        )
        assert upper_neighborhood.shape == (
            int(k_pool * x_0.size(0)),
            int(k_pool * x_0.size(0)),
        )

    def test_reset_parameters(self):
        """Test the reset_parameters method of PoolLayer."""
        k_pool = 0.75
        in_channels_0 = 96
        signal_pool_activation = torch.nn.ReLU()

        # Instantiate the PoolLayer
        pool_layer = PoolLayer(
            in_channels_0=in_channels_0,
            k_pool=k_pool,
            signal_pool_activation=signal_pool_activation,
            readout=True,
        )
        pool_layer.reset_parameters()
        for module in pool_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()


class TestAttentionalLiftLayer:
    """Unit tests for the Attentional Lift class."""

    def test_forward(self):
        """Test the forward method of Attentional Lift."""
        in_channels_0 = 7
        in_channels_1 = 3
        dropout = 0.5
        heads = [1, 3]
        signal_lift_readout = ["cat", "sum", "avg", "max"]
        signal_lift_activation = torch.nn.ReLU()

        n_nodes = 3
        n_edges = n_nodes * n_nodes

        x_0 = torch.randn(n_nodes, in_channels_0)
        x_1 = torch.randn(n_edges, in_channels_1)

        neighborhood = torch.randn(n_nodes, n_nodes)
        neighborhood = neighborhood.to_sparse().float()

        combinations = itertools.product(heads, signal_lift_readout)

        for head, signal_lift_read in combinations:
            can_layer = MultiHeadLiftLayer(
                in_channels_0=in_channels_0,
                heads=head,
                signal_lift_activation=signal_lift_activation,
                signal_lift_dropout=dropout,
                signal_lift_readout=signal_lift_read,
            )
            x_out = can_layer.forward(x_0, neighborhood, x_1)

            if signal_lift_read == "cat":
                assert x_out.shape == (n_edges, head + in_channels_1)
            else:
                assert x_out.shape == (n_edges, 1 + in_channels_1)

    def test_reset_parameters(self):
        """Test the reset_parameters method of Attentional Lift."""
        in_channels = 2

        can_layer = MultiHeadLiftLayer(
            in_channels_0=in_channels,
        )
        can_layer.reset_parameters()

        for module in can_layer.modules():
            if hasattr(module, "reset_parameters"):
                module.reset_parameters()
