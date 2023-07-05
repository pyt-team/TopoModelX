"""Test the template layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.allsettransformer_layer import MLP, AllSetTransformerLayer


class TestAllSetTransformerLayer:
    """Test the template layer."""

    @pytest.fixture
    def allsettransformer_layer(self):
        """Return a template layer."""
        in_dim = 10
        hid_dim = 64
        layer = AllSetTransformerLayer(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            heads=4,
            number_queries=1,
            dropout=0.0,
            mlp_num_layers=1,
            mlp_activation=None,
            mlp_dropout=0.0,
            mlp_norm=None,
        )
        return layer

    def test_forward(self, allsettransformer_layer):
        """Test the forward pass of the template layer."""
        x_0 = torch.randn(3, 10)
        incidence_1 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        output = allsettransformer_layer.forward(x_0, incidence_1)
        assert output.shape == (3, 64)

    def test_forward_with_invalid_input(self, allsettransformer_layer):
        """Test the forward pass of the template layer with invalid input."""
        x_0 = torch.randn(4, 10)
        incidence_1 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        with pytest.raises(ValueError):
            allsettransformer_layer.forward(x_0, incidence_1)

    # def test_MLP(self,):
    #     """Test the MLP class."""
    #     in_channels_ = [10]
    #     hidden_channels_ = [[64], [64, 64]]
    #     norm_layers = [None, torch.nn.LayerNorm]
    #     activation_layers = [torch.nn.ReLU, torch.nn.LeakyReLU]
    #     dropouts = [0.0, 0.5]
    #     bias_ = [True, False]

    #     for in_channels in in_channels_:
    #         for hidden_channels in hidden_channels_:
    #             for norm_layer in norm_layers:
    #                 for activation_layer in activation_layers:
    #                     for dropout in dropouts:
    #                         for bias in bias_:
    #                             mlp = MLP(in_channels=in_channels,
    #                                 hidden_channels=hidden_channels,
    #                                 norm_layer=norm_layer,
    #                                 activation_layer=activation_layer,
    #                                 dropout=dropout,
    #                                 bias=bias)
    #                             _ = mlp(torch.randn(3, in_channels))
