"""Template Layer with two conv passing steps."""
import torch

from topomodelx.base.conv import Conv


class TemplateLayer(torch.nn.Module):
    """Template Layer with two conv passing steps.

    A two-step message passing layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    intermediate_channels : int
        Dimension of intermediate features.
    out_channels : int
        Dimension of output features.
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
    ):
        super().__init__()

        self.conv_level1_1_to_0 = Conv(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )
        self.conv_level2_0_to_1 = Conv(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_1_to_0.reset_parameters()
        self.conv_level2_0_to_1.reset_parameters()

    def forward(self, x_1, incidence_1):
        r"""Forward computation.

        Parameters
        ----------
        x_1 : torch.Tensor, shape=[n_edges, in_channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_1 : torch.Tensor, shape=[n_edges, out_channels]
            Output features on the edges of the simplicial complex.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)
        if x_1.shape[-2] != incidence_1.shape[-1]:
            raise ValueError(
                f"Shape of input face features does not have the correct number of edges {incidence_1.shape[-1]}."
            )
        x_0 = self.conv_level1_1_to_0(x_1, incidence_1)
        x_1 = self.conv_level2_0_to_1(x_0, incidence_1_transpose)
        return x_1
