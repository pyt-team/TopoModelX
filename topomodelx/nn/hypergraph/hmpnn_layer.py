"""HMPNN (Hypergraph Message Passing Neural Network) Layer introduced in Heydari et Livi 2022."""
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from topomodelx.base.message_passing import MessagePassing
from topomodelx.utils.scatter import scatter


class _AdjacencyDropoutMixin:
    training: bool

    def apply_dropout(self, neighborhood, dropout_rate: float):
        neighborhood = neighborhood.coalesce()
        return torch.sparse_coo_tensor(
            neighborhood.indices(),
            F.dropout(
                neighborhood.values().to(torch.float), dropout_rate, self.training
            ),
            neighborhood.size(),
        ).coalesce()


class _NodeToHyperedgeMessenger(MessagePassing, _AdjacencyDropoutMixin):
    def __init__(
        self,
        messaging_func,
        adjacency_dropout: float = 0.7,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
    ) -> None:
        super().__init__(aggr_func)
        self.messaging_func = messaging_func
        self.adjacency_dropout = adjacency_dropout

    def message(self, x_source):
        return self.messaging_func(x_source)

    def forward(self, x_source, neighborhood):
        neighborhood = self.apply_dropout(neighborhood, self.adjacency_dropout)
        source_index_j, self.target_index_i = neighborhood.indices()

        x_message = self.message(x_source)
        x_message_aggregated = self.aggregate(
            x_message.index_select(-2, source_index_j)
        )
        return x_message_aggregated, x_message


class _HyperedgeToNodeMessenger(MessagePassing, _AdjacencyDropoutMixin):
    def __init__(
        self,
        messaging_func,
        adjacency_dropout: float = 0.7,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
    ) -> None:
        super().__init__(aggr_func)
        self.messaging_func = messaging_func
        self.adjacency_dropout = adjacency_dropout

    def message(self, x_source, neighborhood, node_messages):
        hyperedge_neighborhood = self.apply_dropout(
            neighborhood, self.adjacency_dropout
        )
        source_index_j, target_index_i = hyperedge_neighborhood.indices()
        node_messages_aggregated = scatter(self.aggr_func)(
            node_messages.index_select(-2, source_index_j), target_index_i, 0
        )

        return self.messaging_func(x_source, node_messages_aggregated)

    def forward(self, x_source, neighborhood, node_messages):
        x_message = self.message(x_source, neighborhood, node_messages)

        neighborhood = self.apply_dropout(neighborhood, self.adjacency_dropout)
        self.target_index_i, source_index_j = neighborhood.indices()

        x_message_aggregated = self.aggregate(
            x_message.index_select(-2, source_index_j)
        )
        return x_message_aggregated


class _DefaultHyperedgeToNodeMessagingFunc(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        self.linear = nn.Linear(2 * in_features, in_features)

    def forward(self, x_1, m_0):
        return F.sigmoid(self.linear(torch.cat((x_1, m_0), dim=1)))


class _DefaultUpdatingFunc(nn.Module):
    def __init__(self, in_features) -> None:
        super().__init__()
        # self.linear = nn.Linear(in_features, in_features)

    def forward(self, x, m):
        return F.sigmoid(x + m)


class HMPNNLayer(nn.Module):
    r"""HMPNN Layer [1]_.

    The layer is a hypergraph comprised of nodes and hyperedges that makes their new reprsentation using the input
    representation and the messages passed between them. In this layer, the message passed from a node to its
    neighboring hyperedges is only a function of its input representation, but the message from a hyperedge to its
    neighboring nodes is also a function of the messages recieved from them beforehand. This way, a node could have
    a more explicit effect on its upper adjacent neighbors i.e. the nodes that it share a hyperedge with.

    .. math::
        \begin{align*}
        &🟥 \quad m_{{y \rightarrow z}}^{(0 \rightarrow 1)} = M_\mathcal{C} (h_y^{t,(0)}, h_z^{t, (1)})\\
        &🟧 \quad m_{z'}^{(0 \rightarrow 1)} = AGG'{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0\rightarrow1)}\\
        &🟧 \quad m_{z}^{(0 \rightarrow 1)} = AGG_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\\
        &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow0)} = M_\mathcal{B}(h_z^{t,(1)}, m_z^{(1)})\\
        &🟧 \quad m_x^{(1 \rightarrow0)} = AGG_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1 \rightarrow0)}\\
        &🟩 \quad m_x^{(0)} = m_x^{(1 \rightarrow 0)}\\
        &🟩 \quad m_z^{(1)}  = m_{z'}^{(0 \rightarrow 1)}\\
        &🟦 \quad h_x^{t+1, (0)} = U^{(0)}(h_x^{t,(0)}, m_x^{(0)})\\
        &🟦 \quad h_z^{t+1,(1)} = U^{(1)}(h_z^{t,(1)}, m_{z}^{(1)})
        \end{align*}

    Parameters
    ----------
    in_features : int
        Dimension of input features.
    node_to_hyperedge_messaging_func: None
        Node messaging function as a callable or nn.Module object. If not given, a linear plus sigmoid
        function is used, according to the paper.
    hyperedge_to_node_messaging_func: None
        Hyperedge messaging function as a callable or nn.Module object. It gets hyperedge input features
        and aggregated messages of nodes as input and returns hyperedge messages. If not given, two inputs
        are concatenated and a linear layer reducing back to in_features plus sigmoid is applied, according
        to the paper.
    adjacency_dropout: 0.7
        Adjacency dropout rate.
    aggr_func: Literal["sum", "mean", "add"], default="sum"
        Message aggregation function.
    updating_dropout: 0.5
        Regular dropout rate applied to node and hyperedge features.
    updating_func: None
        The final function or nn.Module object to be called on node and hyperedge features to retrieve
        their new representation. If not given, a linear layer is applied, received message is added
        and sigmoid is called.

    References
    ----------
    .. [1] Heydari S, Livi L.
        Message passing neural networks for hypergraphs.
        ICANN 2022.
        https://arxiv.org/abs/2203.16995
    """

    def __init__(
        self,
        in_features,
        node_to_hyperedge_messaging_func=None,
        hyperedge_to_node_messaging_func=None,
        adjacency_dropout: float = 0.7,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        updating_dropout: float = 0.5,
        updating_func=None,
    ) -> None:
        super().__init__()

        if node_to_hyperedge_messaging_func is None:
            node_to_hyperedge_messaging_func = nn.Sequential(
                nn.Linear(in_features, in_features), nn.Sigmoid()
            )
        self.node_to_hyperedge_messenger = _NodeToHyperedgeMessenger(
            node_to_hyperedge_messaging_func, adjacency_dropout, aggr_func
        )
        if hyperedge_to_node_messaging_func is None:
            hyperedge_to_node_messaging_func = _DefaultHyperedgeToNodeMessagingFunc(
                in_features
            )
        self.hyperedge_to_node_messenger = _HyperedgeToNodeMessenger(
            hyperedge_to_node_messaging_func, adjacency_dropout, aggr_func
        )
        self.node_batchnorm = nn.BatchNorm1d(in_features)
        self.hyperedge_batchnorm = nn.BatchNorm1d(in_features)
        self.dropout = torch.distributions.Bernoulli(updating_dropout)

        if updating_func is None:
            updating_func = _DefaultUpdatingFunc(in_features)
        self.updating_func = updating_func

    def apply_regular_dropout(self, x):
        """Apply regular dropout according to the paper.

        Unmasked features in a vector are scaled by d+k / d in which k is the number of
        masked features in the vector and d is the total number of features.
        """
        if self.training:
            mask = self.dropout.sample(x.shape).to(dtype=torch.float, device=x.device)
            d = x.size(0)
            x *= mask * (2 * d - mask.sum(dim=1)).view(-1, 1) / d
        return x

    def forward(self, x_0, x_1, incidence_1):
        r"""Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, node_in_features]
            Input features of the nodes.
        x_1 : torch.Tensor, shape=[n_edges, hyperedge_in_features]
            Input features of the hyperedges.
        incidence_1 : torch.sparse.Tensor
            Incidence matrix mapping hyperedges to nodes (B_1) with shape [n_nodes, n_edges].

        Returns
        -------
        x_0 : torch.Tensor, shape=[n_nodes, node_in_features]
            Output features of the nodes.
        x_1 : torch.Tensor, shape=[n_edges, hyperedge_in_features]
            Output features of the hyperedges.
        """
        node_messages_aggregated, node_messages = self.node_to_hyperedge_messenger(
            x_0, incidence_1
        )
        hyperedge_messages_aggregated = self.hyperedge_to_node_messenger(
            x_1, incidence_1, node_messages
        )

        x_0 = self.updating_func(
            self.apply_regular_dropout(self.node_batchnorm(x_0)),
            hyperedge_messages_aggregated,
        )
        x_1 = self.updating_func(
            self.apply_regular_dropout(self.hyperedge_batchnorm(x_1)),
            node_messages_aggregated,
        )

        return x_0, x_1
