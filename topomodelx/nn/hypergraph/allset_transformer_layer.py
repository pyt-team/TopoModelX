"""AllSetTransformer Layer Module."""
from typing import Literal

import torch
import torch.nn.functional as F
from torch import nn

from topomodelx.base.message_passing import MessagePassing


class AllSetTransformerLayer(nn.Module):
    r"""
    Implementation of the AllSetTransformer Layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    heads : int, default=4
        Number of attention heads.
    number_queries : int, default=1
        Number of queries.
    dropout : float, optional
        Dropout probability.
    mlp_num_layers : int, default=1
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
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=nn.ReLU,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
    ) -> None:
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

    def reset_parameters(self) -> None:
        """Reset parameters."""
        self.vertex2edge.reset_parameters()
        self.edge2vertex.reset_parameters()

    def forward(self, x_0, incidence_1):
        r"""Forward computation.

        Vertex to edge:

        .. math::
            \begin{align*}
                &🟧 \quad m_{\rightarrow z}^{(\rightarrow 1)}
                    = AGG_{y \in \\mathcal{B}(z)} (h_y^{t, (0)}, h_z^{t,(1)}) \quad \text{with attention}\\
                &🟦 \quad h_z^{t+1,(1)}
                    = \text{LN}(m_{\rightarrow z}^{(\rightarrow 1)} + \text{MLP}(m_{\rightarrow z}^{(\rightarrow 1)} ))
            \end{align*}

        Edge to vertex:

        .. math::
            \begin{align*}
                &🟧 \quad m_{\rightarrow x}^{(\rightarrow 0)}
                    = AGG_{z \in \mathcal{C}(x)} (h_z^{t+1,(1)}, h_x^{t,(0)}) \quad \text{with attention}\\
                &🟦 \quad h_x^{t+1,(0)}
                    = \text{LN}(m_{\rightarrow x}^{(\rightarrow 0)} + \text{MLP}(m_{\rightarrow x}^{(\rightarrow 0)} ))
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
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

        x_1 = F.relu(self.vertex2edge(x_0, incidence_1.transpose(1, 0)))
        x_1 = F.dropout(x_1, p=self.dropout, training=self.training)

        x_0 = F.relu(self.edge2vertex(x_1, incidence_1))
        x_0 = F.dropout(x_0, p=self.dropout, training=self.training)

        return x_0, x_1


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
    heads : int, default=4
        Number of attention heads.
    number_queries : int, default=1
        Number of queries.
    dropout : float, default=0.0
        Dropout probability.
    mlp_num_layers : int, default=1
        Number of layers in the MLP.
    mlp_activation : callable or None, optional
        Activation function in the MLP.
    mlp_dropout : float, optional
        Dropout probability in the MLP.
    mlp_norm : str or None, optional
        Type of layer normalization in the MLP.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        heads: int = 4,
        number_queries: int = 1,
        dropout: float = 0.0,
        mlp_num_layers: int = 1,
        mlp_activation=None,
        mlp_dropout: float = 0.0,
        mlp_norm=None,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
    ) -> None:
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

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.multihead_att.reset_parameters()
        self.ln0.reset_parameters()
        self.ln1.reset_parameters()
        if hasattr(self.mlp, "reset_parameters") and callable(
            self.mlp.reset_parameters
        ):
            self.mlp.reset_parameters()

    def forward(self, x_source, neighborhood):
        """Forward computation.

        Parameters
        ----------
        x_source : Tensor, shape = (…, n_source_cells, in_channels)
            Inputer features.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.

        Returns
        -------
        x_message_on_target :
            Output sum over features on target cells.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()

        # Obtain MH from Eq(7) in AllSet paper [1]
        x_message_on_target = self.multihead_att(x_source, neighborhood)

        # Obtain Y from Eq(8) in AllSet paper [1]
        # Skip-connection (broadcased)
        x_message_on_target = x_message_on_target + self.multihead_att.Q_weight

        # Permute: n,h,q,c -> n,q,h,c
        x_message_on_target = x_message_on_target.permute(0, 2, 1, 3)
        x_message_on_target = self.ln0(
            x_message_on_target.reshape(-1, self.number_queries, self.hidden_channels)
        )

        # Obtain LN(Y+FF(Y)) in Eq(8) in AllSet paper [1]
        x_message_on_target = self.ln1(
            x_message_on_target + F.relu(self.mlp(x_message_on_target))
        )

        return x_message_on_target.sum(dim=1)


class MultiHeadAttention(MessagePassing):
    """Computes the multi-head attention mechanism (QK^T)V of transformer-based architectures.

    MH module from Eq(7) in AllSet paper [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    hidden_channels : int
        Dimension of hidden features.
    aggr_norm : bool, default=False
        Whether to normalize the aggregated message by the neighborhood size.
    update_func : str or None, optional
        Update method to apply to message.
    heads : int, default=4
        Number of attention heads.
    number_queries : int, default=1
        Number of queries.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method.
    initialization_gain : float, default=1.414
        Gain factor for initialization.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        aggr_norm: bool = False,
        update_func=None,
        heads: int = 4,
        number_queries: int = 1,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        super().__init__(
            att=True,
            initialization=initialization,
            initialization_gain=initialization_gain,
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

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_uniform_(self.V_weight, gain=self.initialization_gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.K_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.Q_weight, gain=self.initialization_gain)
            torch.nn.init.xavier_normal_(self.V_weight, gain=self.initialization_gain)

        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(self, x_source, neighborhood):
        """Compute (QK^T) of transformer-based architectures.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, heads, number_queries, n_source_cells)
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
        Module MH from Eq (7) in AllSet paper [1]_.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.

        Returns
        -------
        Tensor, shape = (..., n_target_cells, out_channels)
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
        Type of layer normalization.
    activation_layer : callable or None, optional
        Type of activation function.
    dropout : float, optional
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
        activation_layer=torch.nn.ReLU,
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
