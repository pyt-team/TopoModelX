"""Test the HNHN layer."""
import torch
import pytest

from topomodelx.nn.hypergraph.hnhn_layer import HNHNLayer
from topomodelx.base.conv import Conv


class TestHNHNLayer:
    """Test the HNHN layer."""

    @pytest.fixture
    def template_layer(self):
        """Initialize and return an HNHN layer."""
        channels_node = 5
        channels_edge = 8
        n_nodes = 10
        n_edges = 20
        incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()

        return HNHNLayer(
            channels_node=channels_node, 
            channels_edge=channels_edge, 
            incidence_1=incidence_1)
    
    def test_forward(self, template_layer):
        """Test the forward pass of the HNHN layer."""
        n_nodes, n_edges = template_layer.incidence_1.shape
        channels_node = template_layer.channels_node
        channels_edge = template_layer.channels_edge
        x_0 = torch.randn(n_nodes, channels_node)
        x_1 = torch.randn(n_edges, channels_edge)
        x_0_out, x_1_out = template_layer.forward(x_0, x_1)

        assert (x_0_out.shape == x_0.shape)
        assert (x_1_out.shape == x_1.shape)
        return 
    
    def test_compute_normalization_matrices(self, template_layer):
        """Test the computation of the normalization matrices"""
        template_layer.compute_normalization_matrices()

        assert (template_layer.D0_left_alpha_inverse.shape ==\
                (template_layer.n_nodes, template_layer.n_nodes))
        assert (template_layer.D1_left_beta_inverse.shape ==\
                (template_layer.n_edges, template_layer.n_edges))
        assert (template_layer.D1_right_alpha.shape ==\
                (template_layer.n_edges, template_layer.n_edges))
        assert (template_layer.D0_right_beta.shape ==\
                (template_layer.n_nodes, template_layer.n_nodes))
        return
    
    def test_normalize_incidence_matrices(self, template_layer):
        """Test the normalization of the incidence matrices"""
        template_layer.normalize_incidence_matrices()
        
        assert(template_layer.incidence_1.shape ==\
               (template_layer.n_nodes, template_layer.n_edges))
        assert(template_layer.incidence_1_transpose.shape ==\
               (template_layer.n_edges, template_layer.n_nodes))
        return