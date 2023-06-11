"""Test the DHGCN layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.dhgcn_layer import DHGCNLayer


class TestDHGCNLayer:
    """Test the DHGCN layer."""

    @pytest.fixture
    def template_layer(self):
        """Return a DHGCN layer."""
        return DHGCNLayer()

    def test_forward(self, dhgcn_layer):
        """Test the forward pass of the DHGCN layer."""
        pass
