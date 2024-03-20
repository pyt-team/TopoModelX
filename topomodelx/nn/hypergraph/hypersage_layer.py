"""HyperSAGE layer."""
from typing import Literal

import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.message_passing import MessagePassing


class GeneralizedMean(Aggregation):
    """Generalized mean aggregation layer.

    Parameters
    ----------
    power : int, default=2
        Power for the generalized mean.
    **kwargs : keyword arguments, optional
        Arguments for the base aggregation layer.
    """

    def __init__(self, power: int = 2, **kwargs) -> None:
        super().__init__(**kwargs)
        self.power = power

    def forward(self, x: torch.Tensor):
        """Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        torch.Tensor
            Output features.
        """
        n = x.size()[-2]
        x = torch.sum(torch.pow(x, self.power), -2) / n
        return torch.pow(x, 1 / self.power)


class HyperSAGELayer(MessagePassing):
    r"""Implementation of the HyperSAGE layer proposed in [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    out_channels : int
        Dimension of the output features.
    alpha : int, default=-1
        Max number of nodes in a neighborhood to consider. If -1 it considers all the nodes.
    aggr_func_intra : callable, default=GeneralizedMean(p=2)
        Aggregation function. Default is GeneralizedMean(p=2).
    aggr_func_inter : callable, default=GeneralizedMean(p=2)
        Aggregation function. Default is GeneralizedMean(p=2).
    update_func : Literal["relu", "sigmoid"], default="relu"
        Update method to apply to message.
    initialization : Literal["uniform", "xavier_uniform", "xavier_normal"], default="uniform"
        Initialization method.
    device : str, default="cpu"
        Device name to train layer on.

    References
    ----------
    .. [1] Arya, Gupta, Rudinac and Worring.
        HyperSAGE: Generalizing inductive representation learning on hypergraphs (2020).
        https://arxiv.org/abs/2010.04558
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031
    """

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        alpha: int = -1,
        aggr_func_intra: Aggregation | None = None,
        aggr_func_inter: Aggregation | None = None,
        update_func: Literal["relu", "sigmoid"] = "relu",
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "uniform",
        device: str = "cpu",
    ) -> None:
        super().__init__(
            initialization=initialization,
        )

        if aggr_func_intra is None:
            aggr_func_intra = GeneralizedMean(power=2, update_func=None)
        if aggr_func_inter is None:
            aggr_func_inter = GeneralizedMean(power=2, update_func=None)

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.alpha = alpha
        self.aggr_func_intra = aggr_func_intra
        self.aggr_func_inter = aggr_func_inter
        self.update_func = update_func
        self.device = device

        self.weight = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels).to(device=self.device)
        )
        self.reset_parameters()

    def update(self, x_message_on_target: torch.Tensor) -> torch.Tensor:
        r"""Update embeddings on each node (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape = (n_target_nodes, out_channels)
            Output features on target nodes.

        Returns
        -------
        torch.Tensor, shape = (n_target_nodes, out_channels)
            Updated output features on target nodes.
        """
        if self.update_func == "sigmoid":
            return torch.nn.functional.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)
        raise RuntimeError("Update function not recognized.")

    def aggregate(self, x_messages: torch.Tensor, mode: str = "intra"):
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        This function corresponds to either intra- or inter-aggregation.

        Parameters
        ----------
        x_messages : Tensor, shape = (..., n_messages, out_channels)
            Features associated with each message.
            One message is sent from a source cell to a target cell.
        mode : str, default = "inter"
            The mode on which aggregation to compute.
            If set to "inter", will compute inter-aggregation,
            if set to "intra", will compute intra-aggregation (see [1]).

        Returns
        -------
        Tensor, shape = (...,  n_target_cells, out_channels)
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        if mode == "intra":
            return self.aggr_func_intra(x_messages)
        if mode == "inter":
            return self.aggr_func_inter(x_messages)

        raise ValueError(
            "Aggregation mode not recognized. Should be either intra or inter."
        )

    def forward(self, x: torch.Tensor, incidence: torch.Tensor):  # type: ignore[override]
        r"""Forward pass ([2]_ and [3]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow z}^{(0 \rightarrow 1)} = (B_1)^T_{zy} \cdot w_y \cdot (h_y^{(0)})^p\\
            &🟥 \quad m_z^{(0 \rightarrow 1)}  = \left(\frac{1}{\vert \mathcal{B}(z)\vert}\sum_{y \in \mathcal{B}(z)} m_{y \rightarrow z}^{(0 \rightarrow 1)}\right)^{\frac{1}{p}}\\
            &🟥 \quad m_{z \rightarrow x}^{(1 \rightarrow 0)} =  (B_1)_{xz} \cdot w_z  \cdot (m_z^{(0 \rightarrow 1)})^p\\
            &🟧 \quad m_x^{(1,0)}  = \left(\frac{1}{\vert \mathcal{C}(x) \vert}\sum_{z \in \mathcal{C}(x)} m_{z \rightarrow x}^{(1 \rightarrow 0)}\right)^{\frac{1}{p}}\\
            &🟩 \quad m_x^{(0)}  = m_x^{(1 \rightarrow 0)}\\
            &🟦 \quad h_x^{t+1, (0)} = \sigma \left(\frac{m_x^{(0)} + h_x^{t,(0)}}{\lvert m_x^{(0)} + h_x^{t,(0)}\rvert} \cdot \Theta^t\right)
            \end{align*}

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        incidence : torch.Tensor
            Incidence matrix between node/hyperedges.

        Returns
        -------
        torch.Tensor
            Output features.
        """

        def nodes_per_edge(e):
            messages = (
                torch.index_select(
                    input=incidence.to("cpu"), dim=1, index=torch.LongTensor([e])
                )
                .coalesce()
                .indices()[0]
                .to(self.device)
            )
            if len(messages) <= self.alpha or self.alpha == -1:
                return messages
            return messages[torch.randperm(len(messages))[: self.alpha]]

        def edges_per_node(v):
            return (
                torch.index_select(
                    input=incidence.to("cpu"), dim=0, index=torch.LongTensor([v])
                )
                .coalesce()
                .indices()[1]
                .to(self.device)
            )

        messages_per_edges = [
            x[nodes_per_edge(e), :] for e in range(incidence.size()[1])
        ]
        num_of_messages_per_edges = (
            torch.Tensor([message.size()[-2] for message in messages_per_edges])
            .reshape(-1, 1)
            .to(self.device)
        )
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
