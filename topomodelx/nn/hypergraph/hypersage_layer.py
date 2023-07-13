"""HyperSAGE layer."""
import math
from typing import Optional, Tuple, Union

import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.message_passing import MessagePassing


class GeneralizedMean(Aggregation):
    """Generalized mean aggregation layer.

    Parameters
    ----------
    power : int.
        Power for the generalized mean. Default is 2.
    """

    def __init__(self, power: int = 2, **kwargs):
        super().__init__(aggr_func="generalized_mean", **kwargs)
        self.power = power

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
        """
        n = x.size()[-2]
        x = torch.sum(torch.pow(x, self.power), axis=-2) / n
        x = torch.pow(x, 1 / self.power)

        return x


class HyperSAGELayer(MessagePassing):
    r"""Implementation of the HyperSAGE layer proposed in [AGRW20].

    References
    ----------
    .. [AGRW20] Devanshu Arya, Deepak K Gupta, Stevan Rudinac and Marcel Worring. HyperSAGE:
        Generalizing inductive representation learning on hypergraphs. arXiv preprint arXiv:2010.04558. 2020

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    out_channels : int
        Dimension of the output features.
    aggr_func_1: Aggregation
        Aggregation function. Default is GeneralizedMean(p=2).
    aggr_func_2: Aggregation
        Aggregation function. Default is GeneralizedMean(p=2).
    update_func : string
        Update method to apply to message. Default is "relu".
    initialization : string
        Initialization method. Default is "uniform".
    device : string
        Device name to train layer on. Default is "cpu".
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        aggr_func_1: Aggregation = GeneralizedMean(power=2, update_func=None),
        aggr_func_2: Aggregation = GeneralizedMean(power=2, update_func=None),
        update_func: str = "relu",
        initialization: str = "uniform",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.aggr_func_1 = aggr_func_1
        self.aggr_func_2 = aggr_func_2
        self.update_func = update_func
        self.initialization = initialization
        self.device = device

        self.weight = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels).to(device=self.device)
        )
        self.reset_parameters()

    def reset_parameters(self):
        r"""Reset parameters."""
        if self.initialization == "uniform":
            assert self.out_channels > 0, "out_features should be greater than 0"
            stdv = 1.0 / math.sqrt(self.out_channels)
            self.weight.data.uniform_(-stdv, stdv)
        elif self.initialization == "xavier_uniform":
            super().reset_parameters()
        else:
            raise ValueError(
                "Initialization method not recognized. "
                "Should be either uniform or xavier_uniform."
            )

    def update(
        self, x_message_on_target: torch.Tensor, x_target: torch.Tensor = None
    ) -> torch.Tensor:
        r"""Update embeddings on each node (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape=[n_target_nodes, out_channels]
            Output features on target nodes.

        Returns
        -------
        _ : torch.Tensor, shape=[n_target_nodes, out_channels]
            Updated output features on target nodes.
        """
        if self.update_func == "sigmoid":
            return torch.nn.functional.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)

    def aggregate(self, x_messages: torch.Tensor, mode: str = "intra"):
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        🟧 This function corresponds to either intra- or inter-aggregation.

        Parameters
        ----------
        x_messages : Tensor, shape=[..., n_messages, out_channels]
            Features associated with each message.
            One message is sent from a source cell to a target cell.
        mode : string
            The mode on which aggregation to compute. If set to "inter", will compute inter-aggregation,
            if set to "intra", will compute intra-aggregation (see [AGRW20]). Default is "inter".

        Returns
        -------
        _ : Tensor, shape=[...,  n_target_cells, out_channels]
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        if mode == "intra":
            return self.aggr_func_1(x_messages)
        if mode == "inter":
            return self.aggr_func_2(x_messages)
        else:
            raise ValueError(
                "Aggregation mode not recognized.\nShould be either intra or inter."
            )

    def forward(self, x: torch.Tensor, incidence: torch.sparse):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        incidence : torch.sparse
            Incidence matrix between node/hyperedges.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """

        def nodes_per_edge(e):
            return (
                torch.index_select(input=incidence, dim=1, index=torch.LongTensor([e]))
                .coalesce()
                .indices()[0]
            )

        def edges_per_node(v):
            return (
                torch.index_select(input=incidence, dim=0, index=torch.LongTensor([v]))
                .coalesce()
                .indices()[1]
            )

        messages_per_edges = [
            x[nodes_per_edge(e), :] for e in range(incidence.size()[1])
        ]
        num_of_messages_per_edges = torch.Tensor(
            [message.size()[-2] for message in messages_per_edges]
        ).reshape(-1, 1)
        intra_edge_aggregation = torch.stack(
            [self.aggregate(message, mode="intra") for message in messages_per_edges]
        )

        indices_of_edges_per_nodes = [
            edges_per_node(v) for v in range(incidence.size()[0])
        ]
        messages_per_nodes = [
            num_of_messages_per_edges[indices]
            / torch.sum(num_of_messages_per_edges[indices])
            * intra_edge_aggregation[indices, :]
            for indices in indices_of_edges_per_nodes
        ]
        inter_edge_aggregation = torch.stack(
            [self.aggregate(message, mode="inter") for message in messages_per_nodes]
        )

        x_message = x + inter_edge_aggregation

        return self.update(x_message / x_message.norm(p=2) @ self.weight)
