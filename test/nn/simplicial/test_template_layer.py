"""Test the template layer."""
import pytest
import torch

from topomodelx.nn.simplicial.template_layer import TemplateLayer


class TestTemplateLayer:
    """Test the template layer."""

    @pytest.fixture
    def template_layer(self):
        """Return a template layer."""
        in_channels = 10
        intermediate_channels = 20
        out_channels = 30
        return TemplateLayer(in_channels, intermediate_channels, out_channels)

    def test_forward(self, template_layer):
        """Test the forward pass of the template layer."""
        x_2 = torch.randn(3, 10)
        incidence_2 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        output = template_layer.forward(x_2, incidence_2)
        assert output.shape == (3, 30)

    def test_forward_with_invalid_input(self, template_layer):
        """Test the forward pass of the template layer with invalid input."""
        x_2 = torch.randn(4, 10)
        incidence_2 = torch.tensor([[1, 0, 1], [0, 1, 1]], dtype=torch.float32)
        try:
            template_layer.forward(x_2, incidence_2)
            assert False, "Expected ValueError to be raised."
        except ValueError as e:
            assert (
                str(e)
                == f"Shape of input face features does not have the correct number of faces {incidence_2.shape[-1]}."
            )
