"""Test the HyperGAT layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.hypergat_layer import HyperGATLayer


class TestHyperGATLayer:
    """Test the HyperGAT layer."""

    @pytest.fixture
    def hypergat_layer(self):
        """Return a hypergat layer."""
        in_channels = 10
        out_channels = 30
        return HyperGATLayer(in_channels, out_channels)

    def test_forward(self, hypergat_layer):
        """Test the forward pass of the hypergat layer."""
        x_2 = torch.randn(3, 10)
        incidence_2 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        output = hypergat_layer.forward(x_2, incidence_2)
        assert output.shape == (3, 30)

    def test_forward_with_invalid_input(self, hypergat_layer):
        """Test the forward pass of the hypergat layer with invalid input."""
        x_2 = torch.randn(4, 10)
        incidence_2 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        with pytest.raises(RuntimeError):
            hypergat_layer.forward(x_2, incidence_2)

    def test_reset_parameters_xavier_uniform(self, hypergat_layer):
        """Test the reset_parameters method of the HyperSAGE layer with "xavier_uniform" initialization."""
        hypergat_layer.reset_parameters()
        assert hypergat_layer.weight1.requires_grad
        assert hypergat_layer.weight2.requires_grad
        assert hypergat_layer.att_weight1.requires_grad
        assert hypergat_layer.att_weight2.requires_grad

    def test_reset_parameters_xavier_normal(self, hypergat_layer):
        """Test the reset_parameters method of the HyperSAGE layer with "xavier_normal" initialization."""
        hypergat_layer.initialization = "xavier_normal"
        hypergat_layer.reset_parameters()
        assert hypergat_layer.weight1.requires_grad
        assert hypergat_layer.weight2.requires_grad
        assert hypergat_layer.att_weight1.requires_grad
        assert hypergat_layer.att_weight2.requires_grad

    def test_reset_parameters_invalid_initialization(self, hypergat_layer):
        """Test the reset_parameters method of the HyperSAGE layer with invalid initialization."""
        hypergat_layer.initialization = "invalid"
        with pytest.raises(ValueError):
            hypergat_layer.reset_parameters()

    def test_update_relu(self, hypergat_layer):
        """Test the update function with update_func = "relu"."""
        inputs = torch.randn(10, 20)
        updated = hypergat_layer.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, 20)

    def test_update_sigmoid(self, hypergat_layer):
        """Test the update function with update_func = "sigmoid"."""
        hypergat_layer.update_func = "sigmoid"
        inputs = torch.randn(10, 20)
        updated = hypergat_layer.update(inputs)
        assert torch.is_tensor(updated)
        assert updated.shape == (10, 20)

    def test_attention_node_level(self, hypergat_layer):
        """Test the attention function with node-level mechanism."""
        x = torch.randn(3, 30)
        incidence = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        (
            hypergat_layer.target_index_i,
            hypergat_layer.source_index_j,
        ) = incidence.indices()
        output = hypergat_layer.attention(x)
        assert output.shape == (6, 1)

    def test_attention_edge_level(self, hypergat_layer):
        """Test the attention function with edge-level mechanism."""
        x = torch.randn(3, 30)
        incidence = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        (
            hypergat_layer.target_index_i,
            hypergat_layer.source_index_j,
        ) = incidence.indices()
        output = hypergat_layer.attention(x, mechanism="edge-level")
        assert output.shape == (6, 1)
