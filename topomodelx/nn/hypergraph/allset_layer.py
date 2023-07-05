"""AllSet Layer Module."""
import torch.nn.functional as F
from torch import nn

from topomodelx.base.conv import Conv


class AllSetLayer(nn.Module):
    r"""AllSet Layer Module.

    A module for AllSet layer in a bipartite graph.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    dropout : float, optional
        Dropout probability. Default is 0.2.
    mlp_num_layers : int, optional
        Number of layers in the MLP. Default is 2.
    mlp_activation : callable or None, optional
        Activation function in the MLP. Default is None.
    mlp_dropout : float, optional
        Dropout probability in the MLP. Default is 0.0.
    mlp_norm : str or None, optional
        Type of layer normalization in the MLP. Default is None.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout=0.2,
        mlp_num_layers=2,
        mlp_activation=None,
        mlp_dropout=0.0,
        mlp_norm=None,
    ):
        super().__init__()

        self.dropout = dropout

        self.vertex2edge = AllSetBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

        self.edge2vertex = AllSetBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

    def forward(self, x, incidence_1):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor, shape=[n_nodes, channels]
            Node input features.
        incidence_1 : torch.sparse, shape=[n_nodes, n_hyperedges]
            Incidence matrix :math:`B_1` mapping hyperedges to nodes.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """
        x = self.vertex2edge(x, incidence_1.transpose(1, 0))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = self.edge2vertex(x, incidence_1)
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class AllSetBlock(nn.Module):
    """AllSet Block Module.

    A module for AllSet block in a bipartite graph.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    dropout : float, optional
        Dropout probability. Default is 0.2.
    mlp_num_layers : int, optional
        Number of layers in the MLP. Default is 2.
    mlp_activation : callable or None, optional
        Activation function in the MLP. Default is None.
    mlp_dropout : float, optional
        Dropout probability in the MLP. Default is 0.0.
    mlp_norm : callable or None, optional
        Type of layer normalization in the MLP. Default is None.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout,
        mlp_num_layers=2,
        mlp_activation=None,
        mlp_dropout=0.0,
        mlp_norm=None,
    ):
        super(AllSetBlock, self).__init__()

        self.dropout = dropout
        if mlp_num_layers > 0:
            mlp_hidden_channels = [hidden_channels] * mlp_num_layers
            self.encoder = MLP(
                in_channels,
                mlp_hidden_channels,
                norm_layer=mlp_norm,
                activation_layer=mlp_activation,
                dropout=mlp_dropout,
            )
            self.decoder = MLP(
                hidden_channels,
                mlp_hidden_channels,
                norm_layer=mlp_norm,
                activation_layer=mlp_activation,
                dropout=mlp_dropout,
            )
            in_channels = hidden_channels
        else:
            self.encoder = nn.Identity()
            self.decoder = nn.Identity()

        self.conv = Conv(
            in_channels=in_channels,
            out_channels=hidden_channels,
            aggr_norm=True,
            update_func="relu",
            att=False,
        )

    def reset_parameters(self):
        """Reset learnable parameters."""
        self.conv.reset_parameters()
        if not (self.encoder.__class__.__name__ != "Identity"):
            self.encoder.reset_parameters()
        if not (self.decoder.__class__.__name__ != "Identity"):
            self.decoder.reset_parameters()

    def forward(self, x, incidence):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        incidence : torch.sparse
            Incidence matrix between node/hyperedges

        Returns
        -------
        x : torch.Tensor
            Output features.
        """
        x = F.relu(self.encoder(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, incidence)
        x = F.relu(self.decoder(x))

        return x


class MLP(nn.Sequential):
    """MLP Module.

    A module for a multi-layer perceptron (MLP).

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : list of int
        List of dimensions of the hidden features.
    norm_layer : callable or None, optional
        Type of layer normalization. Default is None.
    activation_layer : callable or None, optional
        Type of activation function. Default is None.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    inplace : bool, optional
        Whether to do the operation in-place. Default is False.
    bias : bool, optional
        Whether to add bias. Default is True.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        norm_layer=None,
        activation_layer=None,
        dropout=0.0,
        inplace=None,
        bias=False,
    ):
        params = {} if inplace is None else {"inplace": inplace}
        layers = []
        in_dim = in_channels
        for hidden_dim in hidden_channels[:-1]:
            layers.append(nn.Linear(in_dim, hidden_dim, bias=bias))
            if norm_layer is not None:
                layers.append(norm_layer(hidden_dim))
            layers.append(activation_layer(**params))
            layers.append(nn.Dropout(dropout, **params))
            in_dim = hidden_dim

        layers.append(nn.Linear(in_dim, hidden_channels[-1], bias=bias))
        layers.append(nn.Dropout(dropout, **params))

        super().__init__(*layers)
