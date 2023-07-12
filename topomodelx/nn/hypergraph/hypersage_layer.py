"""HyperSAGE layer."""
import torch

from topomodelx.base.message_passing import MessagePassing


class HyperSAGELayer(MessagePassing):
    """Implementation of the HyperSAGE layer proposed in [DWLLL20].

    References
    ----------
    .. [AGRW20] Devanshu Arya, Deepak K Gupta, Stevan Rudinac and Marcel Worring. HyperSAGE:
        Generalizing inductive representation learning on hypergraphs. arXiv preprint arXiv:2010.04558. 2020
    """

    def __init__(
        self,
        in_features,
        out_features,
        p=2,
        update_func="relu",
        initialization="xavier_uniform",
    ) -> None:
        super().__init__(initialization=initialization)

        self.in_features = in_features
        self.out_features = out_features
        self.p = p
        self.update_func = update_func

        self.weight = torch.nn.Parameter(
            torch.Tensor(self.in_features, self.out_features)
        )
        self.reset_parameters()

    def update(self, x_message_on_target, x_target=None):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x_message_on_target : torch.Tensor, shape=[n_target_cells, out_channels]
            Output features on target cells.

        Returns
        -------
        _ : torch.Tensor, shape=[n_target_cells, out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x_message_on_target)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x_message_on_target)

    def forward(self, x, incidence):
        """Forward pass."""
        if x.shape[-2] != incidence.shape[-2]:
            raise ValueError(
                f"Shape of incidence matrix ({incidence.shape}) does not have the correct number of nodes ({x.shape[0]})."
            )
        intra_edge_aggregation = incidence.transpose(-2, -1) @ torch.pow(x, self.p)
        edges_per_node = (
            lambda v: torch.index_select(
                input=incidence, dim=0, index=torch.LongTensor([v])
            )
            .coalesce()
            .indices()[1]
        )
        global_nbhd = (
            lambda v: torch.index_select(
                input=incidence,
                dim=1,
                index=edges_per_node(v),
            )
            .coalesce()
            .values()
            .size()[0]
        )
        intra_edge_aggregation_scale = torch.Tensor(
            [1.0 / global_nbhd(v) for v in range(x.shape[0])]
        ).reshape(-1, 1)
        inter_edge_aggregation = intra_edge_aggregation_scale * (
            incidence @ intra_edge_aggregation
        )
        inter_edge_aggregation_scale = torch.Tensor(
            [1.0 / edges_per_node(v).shape[0] for v in range(x.shape[0])]
        ).reshape(-1, 1)
        message = torch.pow(
            inter_edge_aggregation_scale * inter_edge_aggregation, 1 / self.p
        )
        return self.update(message / message.norm(p=2) @ self.weight)
