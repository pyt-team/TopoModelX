"""Test the SCCNN layer."""

import torch

from topomodelx.nn.simplicial.sccnn_layer import SCCNNLayer


class TestSCCNNLayer:
    """Test the SCCNN layer."""

    def test_forward(self):
        """Test the forward pass of the SCCNN layer."""
        channels = 5
        n_nodes = 10
        n_edges = 30
        n_faces = 20
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        incidence_2 = torch.randint(0, 2, (n_edges, n_faces)).float()
        laplacian_0 = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        laplacian_down_1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        laplacian_up_1 = torch.randint(0, 2, (n_edges, n_edges)).float()
        laplacian_2 = torch.randint(0, 2, (n_faces, n_faces)).float()
        x_0 = torch.randn(n_nodes, channels)
        x_1 = torch.randn(n_edges, channels)
        x_2 = torch.randn(n_faces, channels)

        x_all = (x_0, x_1, x_2)
        channels_all = (channels, channels, channels)
        laplacian_all = (laplacian_0, laplacian_down_1, laplacian_up_1, laplacian_2)
        incidence_all = (incidence_1, incidence_2)

        # Without aggregation norm, without update function
        sccnn = SCCNNLayer(
            in_channels=channels_all,
            out_channels=channels_all,
            conv_order=2,
            sc_order=2,
            aggr_norm=False,
            update_func=None,
        )
        output = sccnn.forward(x_all, laplacian_all, incidence_all)
        y_0, y_1, y_2 = output
        assert y_0.shape == (n_nodes, channels)
        assert y_1.shape == (n_edges, channels)
        assert y_2.shape == (n_faces, channels)

        # Without aggregation norm, with update function
        sccnn = SCCNNLayer(
            in_channels=channels_all,
            out_channels=channels_all,
            conv_order=2,
            sc_order=2,
            aggr_norm=False,
            update_func="sigmoid",
        )
        output = sccnn.forward(x_all, laplacian_all, incidence_all)
        y_0, y_1, y_2 = output
        assert y_0.shape == (n_nodes, channels)
        assert y_1.shape == (n_edges, channels)
        assert y_2.shape == (n_faces, channels)

        # With aggregation norm, with update function
        sccnn = SCCNNLayer(
            in_channels=channels_all,
            out_channels=channels_all,
            conv_order=2,
            sc_order=2,
            aggr_norm=True,
            update_func="sigmoid",
        )
        output = sccnn.forward(x_all, laplacian_all, incidence_all)
        y_0, y_1, y_2 = output
        assert y_0.shape == (n_nodes, channels)
        assert y_1.shape == (n_edges, channels)
        assert y_2.shape == (n_faces, channels)
