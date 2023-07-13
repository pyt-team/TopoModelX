"""HyperSAGE layer."""
import math

import torch

from topomodelx.base.message_passing import MessagePassing


class HyperSAGELayer(MessagePassing):
    r"""Implementation of the HyperSAGE layer proposed in [DWLLL20].

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
    p : int.
        Power for the generalized mean in the aggregation. Default is 2.
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
        p: int = 2,
        update_func: str = "relu",
        initialization: str = "uniform",
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p = p
        self.update_func = update_func
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
            self.weight.uniform_(-stdv, stdv)
        if self.initialization == "xavier_uniform":
            super().reset_parameters()
        else:
            raise RuntimeError(
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
        intra_edge_aggregation_scale = (
            torch.Tensor([1.0 / global_nbhd(v) for v in range(x.shape[0])])
            .reshape(-1, 1)
            .to(device=self.device)
        )
        inter_edge_aggregation = intra_edge_aggregation_scale * (
            incidence @ intra_edge_aggregation
        )
        inter_edge_aggregation_scale = (
            torch.Tensor([1.0 / edges_per_node(v).shape[0] for v in range(x.shape[0])])
            .reshape(-1, 1)
            .to(device=self.device)
        )
        message = torch.pow(
            inter_edge_aggregation_scale * inter_edge_aggregation, 1.0 / self.p
        )
        return self.update(message / message.norm(p=2) @ self.weight)
