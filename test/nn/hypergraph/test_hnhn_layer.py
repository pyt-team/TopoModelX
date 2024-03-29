"""Test the HNHN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.hnhn_layer import HNHNLayer


class TestHNHNLayer:
    """Class to support HNHNLayer testing."""

    @pytest.fixture
    def template_layer(self):
        """Initialize and return an HNHN layer."""
        self.in_channels = 5
        self.hidden_channels = 8
        n_nodes = 10
        n_edges = 20
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()

        return HNHNLayer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            incidence_1=incidence_1,
        )

    @pytest.fixture
    def template_layer2(self):
        """Initialize and return an HNHN layer."""
        self.in_channels = 5
        self.hidden_channels = 8

        return HNHNLayer(
            in_channels=self.in_channels,
            hidden_channels=self.hidden_channels,
            incidence_1=None,
            bias_init="xavier_normal",
        )

    def test_forward(self, template_layer, template_layer2):
        """Test the forward pass of the HNHN layer."""
        n_nodes, n_edges = template_layer.incidence_1.shape

        x_0 = torch.randn(n_nodes, self.in_channels)
        x_0_out, x_1_out = template_layer.forward(x_0)

        assert x_0_out.shape == (n_nodes, self.hidden_channels)
        assert x_1_out.shape == (n_edges, self.hidden_channels)

        n_nodes = 10
        n_edges = 20
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()

        x_0_out, x_1_out = template_layer2.forward(x_0, incidence_1)

        assert x_0_out.shape == (n_nodes, self.hidden_channels)
        assert x_1_out.shape == (n_edges, self.hidden_channels)

        return

    def test_compute_normalization_matrices(self, template_layer):
        """Test the computation of the normalization matrices."""
        template_layer.compute_normalization_matrices()

        assert template_layer.D0_left_alpha_inverse.shape == (
            template_layer.n_nodes,
            template_layer.n_nodes,
        )
        assert template_layer.D1_left_beta_inverse.shape == (
            template_layer.n_edges,
            template_layer.n_edges,
        )
        assert template_layer.D1_right_alpha.shape == (
            template_layer.n_edges,
            template_layer.n_edges,
        )
        assert template_layer.D0_right_beta.shape == (
            template_layer.n_nodes,
            template_layer.n_nodes,
        )
        return

    def test_normalize_incidence_matrices(self, template_layer):
        """Test the normalization of the incidence matrices."""
        template_layer.normalize_incidence_matrices()

        assert template_layer.incidence_1.shape == (
            template_layer.n_nodes,
            template_layer.n_edges,
        )
        assert template_layer.incidence_1_transpose.shape == (
            template_layer.n_edges,
            template_layer.n_nodes,
        )
        return

    def test_reset_parameters(self, template_layer):
        """Test reset parameters."""
        shape_1_to_0_in = template_layer.conv_1_to_0.weight.shape
        shape_0_to_1_in = template_layer.conv_0_to_1.weight.shape
        template_layer.reset_parameters()
        shape_1_to_0_out = template_layer.conv_1_to_0.weight.shape
        shape_0_to_1_out = template_layer.conv_0_to_1.weight.shape
        assert shape_1_to_0_in == shape_1_to_0_out
        assert shape_0_to_1_in == shape_0_to_1_out

    def check_bias_type(self, template_layer):
        """Check bias initialization type."""
        assert template_layer.bias_init in ["xavier_uniform", "xavier_normal"]
