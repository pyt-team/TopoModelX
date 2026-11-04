"""Test the SCCNN layer."""

import pytest
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

    @pytest.mark.parametrize("sc_order", [2, 3])
    def test_identity_terms_and_gradients(self, sc_order):
        """Preserve all seven identity terms and their training gradients."""
        torch.manual_seed(42)
        layer = SCCNNLayer((2, 2, 2), (2, 2, 2), conv_order=1, sc_order=sc_order)
        x_0, x_1, x_2 = [torch.randn(n, 2, requires_grad=True) for n in (3, 3, 1)]
        # Signed boundaries of one triangle.
        b1 = torch.tensor([[-1.0, -1.0, 0.0], [1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
        b2 = torch.tensor([[1.0], [-1.0], [1.0]])

        # Zero filters isolate the identity terms we changed.
        laplacians = [torch.zeros(n, n).to_sparse() for n in (3, 3, 3, 1)]
        if sc_order > 2:
            laplacians.append(torch.zeros(1, 1).to_sparse())
        actual = layer((x_0, x_1, x_2), laplacians, (b1.to_sparse(), b2.to_sparse()))

        # Identity signals occupy these slots in the original filter stacks.
        w0, w1, w2 = layer.weight_0, layer.weight_1, layer.weight_2
        face_transfer_slot = 2 if sc_order == 2 else 3
        expected = (
            x_0 @ w0[:, :, 0] + (b1 @ x_1) @ w0[:, :, 2],
            (b1.T @ x_0) @ w1[:, :, 0] + x_1 @ w1[:, :, 2] + (b2 @ x_2) @ w1[:, :, 5],
            x_2 @ w2[:, :, 0] + (b2.T @ x_1) @ w2[:, :, face_transfer_slot],
        )
        for result, reference in zip(actual, expected, strict=True):
            torch.testing.assert_close(result, reference)

        # Compare gradients for both input features and learned weights.
        variables = (x_0, x_1, x_2, w0, w1, w2)
        actual_grad = torch.autograd.grad(
            sum(y.square().sum() for y in actual), variables
        )
        expected_grad = torch.autograd.grad(
            sum(y.square().sum() for y in expected), variables
        )
        for result, reference in zip(actual_grad, expected_grad, strict=True):
            torch.testing.assert_close(result, reference)
