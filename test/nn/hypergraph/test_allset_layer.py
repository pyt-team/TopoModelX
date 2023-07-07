"""Test the AllSet layer."""
import pytest
import torch

from topomodelx.nn.hypergraph.allset_layer import MLP, AllSetLayer, AllSetBlock


class TestAllSetLayer:
    """Test the AllSet layer."""

    @pytest.fixture
    def allset_layer(self):
        """Return a AllSet layer."""
        in_dim = 10
        hid_dim = 64
        layer = AllSetLayer(
            in_channels=in_dim,
            hidden_channels=hid_dim,
            dropout=0.0,
            mlp_num_layers=1,
            mlp_activation=None,
            mlp_dropout=0.0,
            mlp_norm=None,
        )
        return layer

    def test_forward(self, allset_layer):
        """Test the forward pass of the AllSet layer."""
        x_0 = torch.randn(3, 10)
        incidence_1 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        output = allset_layer.forward(x_0, incidence_1)
        assert output.shape == (3, 64)

    def test_AllSetBlock(self):
        """Test the AllSetBlock class.

        (used in AllSetLayer)
        """
        in_channels = 10
        hidden_channels = 64
        mlp_num_layers = [0, 1, 2]
        dropout = 0.0
        for mlp in mlp_num_layers:
            allset_block = AllSetBlock(
                in_channels=in_channels,
                hidden_channels=hidden_channels,
                mlp_num_layers=mlp,
                dropout=dropout,
            )
            x_0 = torch.randn(3, 10)
            incidence_1 = torch.tensor(
                [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
            ).to_sparse()
            output = allset_block.forward(x_0, incidence_1)
            assert output.shape == (3, 64)

    def test_forward_with_invalid_input(self, allset_layer):
        """Test the forward pass of the AllSet layer with invalid input."""
        x_0 = torch.randn(4, 10)
        incidence_1 = torch.tensor(
            [[1, 0, 0], [0, 1, 1], [1, 1, 1]], dtype=torch.float32
        ).to_sparse()
        with pytest.raises(ValueError):
            allset_layer.forward(x_0, incidence_1)

    def test_initialisation_mlp_num_layers_zero(self):
        """Test the initialisation of the AllSet layer with invalid input."""
        with pytest.raises(ValueError):
            mlp_num_layers = 0
            _ = AllSetLayer(
                in_channels=10,
                hidden_channels=64,
                mlp_num_layers=mlp_num_layers,
            )

    def reset_parameters(self, allset_layer):
        """Test the reset_parameters method of the AllSet layer."""
        allset_layer.reset_parameters()
        if not (allset_layer.vertex2edge.encoder.__class__.__name__ != "Identity"):
            assert allset_layer.vertex2edge.encoder.weight.requires_grad

        if not (allset_layer.edge2vertex.encoder.__class__.__name__ != "Identity"):
            assert allset_layer.edge2vertex.encoder.weight.requires_grad

    def test_initialisation_mlp_num_layers_negative(self):
        """Test the initialisation of the AllSet layer with invalid input."""
        with pytest.raises(ValueError):
            mlp_num_layers = -1
            _ = AllSetLayer(
                in_channels=10,
                hidden_channels=64,
                mlp_num_layers=mlp_num_layers,
            )

    def test_MLP(self):
        """Test the MLP class.

        (used in AllSetLayer)
        """
        in_channels_ = [10]
        hidden_channels_ = [[64], [64, 64]]
        norm_layers = [None, torch.nn.LayerNorm]
        activation_layers = [torch.nn.ReLU, torch.nn.LeakyReLU]
        dropouts = [0.0, 0.5]
        bias_ = [True, False]

        for in_channels in in_channels_:
            for hidden_channels in hidden_channels_:
                for norm_layer in norm_layers:
                    for activation_layer in activation_layers:
                        for dropout in dropouts:
                            for bias in bias_:
                                mlp = MLP(
                                    in_channels=in_channels,
                                    hidden_channels=hidden_channels,
                                    norm_layer=norm_layer,
                                    activation_layer=activation_layer,
                                    dropout=dropout,
                                    bias=bias,
                                )
                                assert mlp is not None
