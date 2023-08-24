"""AllSetTransformer Layer Module."""
import torch
import torch.nn.functional as F
from torch import nn

from topomodelx.base.message_passing import MessagePassing


class AllSetTransformerLayer(nn.Module):
    r"""
    Implementation of the AllSetTransformer Layer proposed in [ECCP22].

    References
    ----------
    .. [ECCP22] Chien, E., Pan, C., Peng, J., & Milenkovic, O. You are AllSet: A Multiset
      Function Framework for Hypergraph Neural Networks. In International Conference on
      Learning Representations, 2022 (https://arxiv.org/pdf/2106.13264.pdf)

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    out_channels : int
        Dimension of the output features.
    num_heads : int, optional
        Number of attention heads. Default is 4.
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
        heads=4,
        number_queries=1,
        dropout=0.0,
        mlp_num_layers=1,
        mlp_activation=nn.ReLU,
        mlp_dropout=0.0,
        mlp_norm=None,
    ):
        super().__init__()

        if heads <= 0:
            raise ValueError(f"heads ({heads}) must be positive")

        if mlp_num_layers <= 0:
            raise ValueError(f"mlp_num_layers ({mlp_num_layers}) must be positive")

        if (hidden_channels % heads) != 0:
            raise ValueError(
                f"hidden_channels ({hidden_channels}) must be divisible by heads ({heads})"
            )

        self.dropout = dropout

        self.vertex2edge = AllSetTransformerBlock(
            in_channels=in_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

        self.edge2vertex = AllSetTransformerBlock(
            in_channels=hidden_channels,
            hidden_channels=hidden_channels,
            dropout=dropout,
            heads=heads,
            number_queries=number_queries,
            mlp_num_layers=mlp_num_layers,
            mlp_activation=mlp_activation,
            mlp_dropout=mlp_dropout,
            mlp_norm=mlp_norm,
        )

    def reset_parameters(self):
        """Reset parameters."""
        self.vertex2edge.reset_parameters()
        self.edge2vertex.reset_parameters()

    def forward(self, x, incidence_1):
        r"""Forward computation.

        .. math::
            "Following the \"awesome-tnns\" [github repo.](https://github.com/awesome-tnns/awesome-tnns/blob/main/Hypergraphs.md)
            Vertex to edge:
                🟧 $\\quad m_{\\rightarrow z}^{(\\rightarrow 1)}
                    = AGG_{y \\in \\mathcal{B}(z)} (h_y^{t, (0)}, h_z^{t,(1)}) \\quad \\text{with attention}$
                🟦 $\\quad h_z^{t+1,(1)}
                    = \\text{LN}(m_{\\rightarrow z}^{(\\rightarrow 1)} + \\text{MLP}(m_{\\rightarrow z}^{(\\rightarrow 1)} ))$

            Edge to vertex:
                🟧 $\\quad m_{\\rightarrow x}^{(\\rightarrow 0)}
                    = AGG_{z \\in \\mathcal{C}(x)} (h_z^{t+1,(1)}, h_x^{t,(0)}) \\quad \\text{with attention}$
                🟦 $\\quad h_x^{t+1,(0)}
                    = \\text{LN}(m_{\\rightarrow x}^{(\\rightarrow 0)} + \\text{MLP}(m_{\\rightarrow x}^{(\\rightarrow 0)} ))$

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
        if x.shape[-2] != incidence_1.shape[-2]:
            raise ValueError(
                f"Shape of incidence matrix ({incidence_1.shape}) does not have the correct number of nodes ({x.shape[0]})."
            )

        x = F.relu(self.vertex2edge(x, incidence_1.transpose(1, 0)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.edge2vertex(x, incidence_1))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x


class AllSetTransformerBlock(nn.Module):
    r"""
    AllSetTransformer Block Module.

    A module for AllSet Transformer block in a bipartite graph.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    heads : int, optional
        Number of attention heads. Default is 4.
    number_queries : int, optional
        Number of queries. Default is 1.
    dropout : float, optional
        Dropout probability. Default is 0.0.
    mlp_num_layers : int, optional
        Number of layers in the MLP. Default is 1.
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
        heads=4,
        number_queries=1,
        dropout=0.0,
        mlp_num_layers=1,
        mlp_activation=None,
        mlp_dropout=0.0,
        mlp_norm=None,
        initialization="xavier_uniform",
    ):
        super().__init__()

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.heads = heads
        self.number_queries = number_queries
        self.dropout = dropout

        # For neighbor nodes (source side, key)
        self.multihead_att = MultiHeadAttention(
            in_channels=self.in_channels,
            hidden_channels=hidden_channels // heads,
            heads=self.heads,
            number_queries=number_queries,
            initialization=initialization,
        )

        self.mlp = MLP(
            in_channels=self.hidden_channels,
            hidden_channels=[self.hidden_channels] * mlp_num_layers,
            norm_layer=mlp_norm,
            activation_layer=mlp_activation,
            dropout=mlp_dropout,
        )

        self.ln0 = nn.LayerNorm(self.hidden_channels)
        self.ln1 = nn.LayerNorm(self.hidden_channels)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.multihead_att.reset_parameters()
        for layer in self.mlp.children():
            if hasattr(layer, "reset_parameters"):
                layer.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()

    def forward(self, x_source, neighborhood):
        """Forward computation.

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
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()

        # Obtain MH from Eq(7) in AllSet paper [ECCP22]
        x_message_on_target = self.multihead_att(x_source, neighborhood)

        # Obtain Y from Eq(8) in AllSet paper [ECCP22]
        # Skip-connection (broadcased)
        x_message_on_target = x_message_on_target + self.multihead_att.Q_weight

        # Permute: n,h,q,c -> n,q,h,c
        x_message_on_target = x_message_on_target.permute(0, 2, 1, 3)
        x_message_on_target = self.ln0(
            x_message_on_target.reshape(-1, self.number_queries, self.hidden_channels)
        )

        # Obtain LN(Y+FF(Y)) in Eq(8) in AllSet paper [ECCP22]
        x_message_on_target = self.ln1(
            x_message_on_target + F.relu(self.mlp(x_message_on_target))
        )

        return x_message_on_target.sum(dim=1)


class MultiHeadAttention(MessagePassing):
    """Computes the multi-head attention mechanism (QK^T)V of transformer-based architectures.

    MH module from Eq(7) in AllSet paper [ECCP22]

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    hidden_channels : int
        Dimension of hidden features.
    aggr_norm : bool, optional
        Whether to normalize the aggregated message by the neighborhood size. Default is False.
    update_func : str or None, optional
        Update method to apply to message. Default is None.
    heads : int, optional
        Number of attention heads. Default is 4.
    number_queries : int, optional
        Number of queries. Default is 1.
    initialization : str, optional
        Initialization method. Default is "xavier_uniform".
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        aggr_norm=False,
        update_func=None,
        heads=4,
        number_queries=1,
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.in_channels = in_channels
        self.hidden_channels = hidden_channels

        self.aggr_norm = aggr_norm
        self.update_func = update_func

        self.heads = heads
        self.number_queries = number_queries

        self.K_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )
        self.Q_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.number_queries, self.hidden_channels)
        )
        self.V_weight = torch.nn.Parameter(
            torch.randn(self.heads, self.in_channels, self.hidden_channels)
        )

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.K_weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.Q_weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.V_weight, gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.K_weight, gain=gain)
            torch.nn.init.xavier_normal_(self.Q_weight, gain=gain)
            torch.nn.init.xavier_normal_(self.V_weight, gain=gain)

        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(self, x_source, neighborhood):
        """Compute (QK^T) of transformer-based architectures.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        _ : torch.Tensor, shape = [n_target_cells, heads, number_queries, n_source_cells]
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_K = torch.matmul(x_source, self.K_weight)
        alpha = torch.matmul(self.Q_weight, x_K.transpose(1, 2))
        expanded_alpha = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=alpha.permute(*torch.arange(alpha.ndim - 1, -1, -1))[
                self.source_index_j
            ],
            size=[
                neighborhood.shape[0],
                neighborhood.shape[1],
                alpha.shape[1],
                alpha.shape[0],
            ],
        )
        alpha_soft = (
            torch.sparse.softmax(expanded_alpha, dim=1).to_dense().transpose(1, 3)
        )
        return alpha_soft

    def forward(self, x_source, neighborhood):
        """Forward pass.

        Computes (QK^T)V attention mechanism of transformer-based architectures.
        Module MH from Eq(7) in AllSet paper [ECCP22]

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        # Transformer-based attention mechanism
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        attention_values = self.attention(x_source, neighborhood)

        x_message = torch.matmul(x_source, self.V_weight)
        x_message_on_target = torch.matmul(attention_values, x_message)

        return x_message_on_target


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
        activation_layer=torch.nn.ReLU,
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
