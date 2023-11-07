"""AllSet Layer Module."""
import torch.nn.functional as F
from torch import nn

from topomodelx.base.conv import Conv


class AllSetLayer(nn.Module):
    """
    AllSet Layer Module [1]_.

    A module for AllSet layer in a bipartite graph.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    dropout : float, default=0.2
        Dropout probability.
    mlp_num_layers : int, default=2
        Number of layers in the MLP.
    mlp_activation : callable or None, optional
        Activation function in the MLP.
    mlp_dropout : float, optional
        Dropout probability in the MLP.
    mlp_norm : str or None, optional
        Type of layer normalization in the MLP.

    References
    ----------
    .. [1] Chien, Pan, Peng and Milenkovic.
        You are AllSet: a multiset function framework for hypergraph neural networks.
        ICLR 2022.
        https://arxiv.org/abs/2106.13264
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout: float = 0.2,
        mlp_num_layers: int = 2,
        mlp_activation=nn.ReLU,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
    ) -> None:
        super().__init__()

        if mlp_num_layers <= 0:
            raise ValueError(f"mlp_num_layers ({mlp_num_layers}) must be positive")
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

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        self.vertex2edge.reset_parameters()
        self.edge2vertex.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""
        Forward computation.

        Vertex to edge:

        .. math::
            \begin{align*}
                    &🟧 \quad m_{\rightarrow z}^{(\rightarrow 1)}
                        = AGG_{y \in \mathcal{B}(z)} (h_y^{t, (0)}, h_z^{t,(1)}) \\
                    &🟦 \quad h_z^{t+1,(1)}
                        = \sigma(m_{\rightarrow z}^{(\rightarrow 1)})
            \end{align*}

        Edge to vertex:

        .. math::
            \begin{align*}
                    &🟧 \quad m_{\rightarrow x}^{(\rightarrow 0)}
                        = AGG_{z \in \mathcal{C}(x)} (h_z^{t+1,(1)}, h_x^{t,(0)}) \\
                    &🟦 \quad h_x^{t+1,(0)}
                        = \sigma(m_{\rightarrow x}^{(\rightarrow 0)})
            \end{align*}

        Parameters
        ----------
        x : torch.Tensor, shape = (n_nodes, channels)
            Node input features.
        incidence_1 : torch.sparse, shape = (n_nodes, n_hyperedges)
            Incidence matrix :math:`B_1` mapping hyperedges to nodes.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        if x_0.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Shape of incidence matrix ({incidence_1.shape}) does not have the correct number of nodes ({x_0.shape[0]})."
            )

        x_1 = self.vertex2edge(x_0, incidence_1.transpose(1, 0))
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_0 = self.edge2vertex(x_1, incidence_1)
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)

        return (x_0, x_1)


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
        Type of layer normalization.
    activation_layer : callable or None, optional
        Type of activation function.
    dropout : float, default=0.0
        Dropout probability.
    inplace : bool, default=False
        Whether to do the operation in-place.
    bias : bool, default=False
        Whether to add bias.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        norm_layer=None,
        activation_layer=None,
        dropout: float = 0.0,
        inplace: bool | None = None,
        bias: bool = False,
    ) -> None:
        params = {} if inplace is None else {"inplace": inplace}
        layers: list[nn.Module] = []
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


class AllSetBlock(nn.Module):
    """AllSet Block Module.

    A module for AllSet block in a bipartite graph.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    dropout : float, default=0.2
        Dropout probability.
    mlp_num_layers : int, default=2
        Number of layers in the MLP.
    mlp_activation : callable or None, optional
        Activation function in the MLP.
    mlp_dropout : float, optional
        Dropout probability in the MLP.
    mlp_norm : callable or None, optional
        Type of layer normalization in the MLP.
    """

    encoder: MLP | nn.Identity
    decoder: MLP | nn.Identity

    def __init__(
        self,
        in_channels,
        hidden_channels,
        dropout: float = 0.2,
        mlp_num_layers: int = 2,
        mlp_activation=nn.ReLU,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
    ) -> None:
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

    def reset_parameters(self) -> None:
        """Reset learnable parameters."""
        if callable(self.encoder.reset_parameters):
            self.encoder.reset_parameters()
        if callable(self.decoder.reset_parameters):
            self.decoder.reset_parameters()
        self.conv.reset_parameters()

    def forward(self, x, incidence):
        """
        Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input node features.
        incidence : torch.sparse
            Incidence matrix between node/hyperedges.

        Returns
        -------
        torch.Tensor
            Output features.
        """
        x = F.relu(self.encoder(x))
        x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.conv(x, incidence)
        x = F.relu(self.decoder(x))

        return x
