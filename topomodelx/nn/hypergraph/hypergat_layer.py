"""HyperGAT layer."""
from typing import Literal

import torch

from topomodelx.base.message_passing import MessagePassing


class HyperGATLayer(MessagePassing):
    r"""Implementation of the HyperGAT layer proposed in [DWLLL20].

    References
    ----------
    .. [DWLLL20] Kaize Ding, Jianling Wang, Jundong Li, Dingcheng Li, & Huan Liu. Be more with less:
        Hypergraph attention networks for inductive text classification. In Proceedings of the 2020 Conference
        on Empirical Methods in Natural Language Processing (EMNLP), 2020 (https://aclanthology.org/2020.emnlp-main.399.pdf)

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    out_channels : int
        Dimension of the output features.
    update_func : string
        Update method to apply to message. Default is "relu".
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        update_func: str = "relu",
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        super().__init__(initialization=initialization)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.update_func = update_func

        self.weight1 = torch.nn.Parameter(
            torch.Tensor(self.in_channels, self.out_channels)
        )
        self.weight2 = torch.nn.Parameter(
            torch.Tensor(self.out_channels, self.out_channels)
        )

        self.att_weight1 = torch.nn.Parameter(torch.zeros(size=(out_channels, 1)))
        self.att_weight2 = torch.nn.Parameter(torch.zeros(size=(2 * out_channels, 1)))
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        r"""Reset parameters."""
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight1, gain=gain)
            torch.nn.init.xavier_uniform_(self.weight2, gain=gain)
            torch.nn.init.xavier_uniform_(self.att_weight1.view(-1, 1), gain=gain)
            torch.nn.init.xavier_uniform_(self.att_weight2.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight1, gain=gain)
            torch.nn.init.xavier_normal_(self.weight2, gain=gain)
            torch.nn.init.xavier_normal_(self.att_weight1.view(-1, 1), gain=gain)
            torch.nn.init.xavier_normal_(self.att_weight2.view(-1, 1), gain=gain)
        else:
            raise ValueError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def attention(
        self,
        x_source,
        x_target=None,
        mechanism: Literal["node-level", "edge-level"] = "node-level",
    ):
        r"""Compute attention weights for messages, as proposed in [DWLLL20].

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape=[n_target_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        mechanism: Literal["node-level", "edge-level"]
            Attention mechanism as proposed in [DWLLL20]. If set to "node-level", will compute node-level attention,
            if set to "edge-level", will compute edge-level attention (see [DWLLL20]). Default is "node-level".

        Returns
        -------
        torch.Tensor, shape = [n_messages, 1]
            Attention weights: one scalar per message between a source and a target cell.
        """
        if mechanism == "node-level":
            x_source_per_message = x_source[self.target_index_i]
            return torch.nn.functional.softmax(
                torch.matmul(
                    torch.nn.functional.leaky_relu(x_source_per_message),
                    self.att_weight1,
                ),
                dim=1,
            )

        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.nn.functional.leaky_relu(
            torch.cat([x_source_per_message, x_target_per_message], dim=1)
        )
        return torch.nn.functional.softmax(
            torch.matmul(x_source_target_per_message, self.att_weight2), dim=1
        )

    def update(self, x_message_on_target, x_target=None):
        r"""Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape=[n_target_cells, out_channels]
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape=[n_target_cells, out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)

    def forward(self, x_source, incidence):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        incidence : torch.sparse
            Incidence matrix between nodes and hyperedges.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """
        intra_aggregation = incidence.t() @ (x_source @ self.weight1)

        self.target_index_i, self.source_index_j = incidence.indices()

        attention_values = self.attention(intra_aggregation).squeeze()
        incidence_with_attention = torch.sparse_coo_tensor(
            indices=incidence.indices(),
            values=incidence.values() * attention_values,
            size=incidence.shape,
        )
        intra_aggregation_with_attention = incidence_with_attention.t() @ (
            x_source @ self.weight1
        )
        messages_on_edges = self.update(intra_aggregation_with_attention)

        inter_aggregation = incidence @ (messages_on_edges @ self.weight2)

        attention_values = self.attention(
            inter_aggregation, intra_aggregation
        ).squeeze()
        incidence_with_attention = torch.sparse_coo_tensor(
            indices=incidence.indices(),
            values=attention_values * incidence.values(),
            size=incidence.shape,
        )
        inter_aggregation_with_attention = incidence_with_attention @ (
            messages_on_edges @ self.weight2
        )
        return self.update(inter_aggregation_with_attention)
