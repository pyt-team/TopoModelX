"""Test the SCNN layer."""

import torch

from topomodelx.nn.simplicial.scnn_layer import SCNNLayer


class TestSCNNLayer:
    """Test the SCNN layer."""

    def test_forward(self):
        """Test the forward pass of the SCNN layer."""
        in_channels = 5
        out_channels = 5
        conv_order_down = 3
        conv_order_up = 3

        n_simplices = 10
        laplacian_down = torch.randint(0, 2, (n_simplices, n_simplices)).float()
        laplacian_up = torch.randint(0, 2, (n_simplices, n_simplices)).float()
        x = torch.randn(n_simplices, in_channels)

        # Test 1: Without aggregation normalization, without update function
        scnn = SCNNLayer(
            in_channels,
            out_channels,
            conv_order_down,
            conv_order_up,
            aggr_norm=False,
            update_func=None,
        )
        output = scnn.forward(
            x, laplacian_down=laplacian_down, laplacian_up=laplacian_up
        )

        assert output.shape == (n_simplices, out_channels)

        # Test 2: Without aggregation normalization, With update function
        scnn = SCNNLayer(
            in_channels,
            out_channels,
            conv_order_down,
            conv_order_up,
            aggr_norm=False,
            update_func="sigmoid",
        )
        output = scnn.forward(
            x, laplacian_down=laplacian_down, laplacian_up=laplacian_up
        )

        assert output.shape == (n_simplices, out_channels)

        # Test 3: With aggregation normalization, with Relu update function
        scnn = SCNNLayer(
            in_channels,
            out_channels,
            conv_order_down,
            conv_order_up,
            aggr_norm=True,
            update_func="relu",
        )
        output = scnn.forward(
            x, laplacian_down=laplacian_down, laplacian_up=laplacian_up
        )

        assert output.shape == (n_simplices, out_channels)
        # Test 4: conv_order_down is 0 and conv_order_up is not
        scnn = SCNNLayer(
            in_channels,
            out_channels,
            conv_order_down=0,
            conv_order_up=conv_order_up,
        )
        output = scnn.forward(
            x, laplacian_down=laplacian_down, laplacian_up=laplacian_up
        )

        assert output.shape == (n_simplices, out_channels)

        # Test 5: conv_order_down is 3 and conv_order_up is 0
        scnn = SCNNLayer(
            in_channels,
            out_channels,
            conv_order_down=conv_order_down,
            conv_order_up=0,
        )
        output = scnn.forward(
            x, laplacian_down=laplacian_down, laplacian_up=laplacian_up
        )

        assert output.shape == (n_simplices, out_channels)
