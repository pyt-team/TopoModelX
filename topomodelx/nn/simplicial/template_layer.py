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
    out_channels : int
        Dimension of output features.
    aggr_func : string
        Aggregation method.
        (Inter-neighborhood).
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
    ):
        super().__init__()

        self.conv_level1_2_to_1 = Conv(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )
        self.conv_level2_1_to_2 = Conv(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_2_to_1.reset_parameters()
        self.conv_level2_1_to_2.reset_parameters()

    def forward(self, x_2, incidence_2):
        r"""Forward computation.

        Parameters
        ----------
        x_2 : torch.tensor, shape=[n_faces, in_channels]
            Input features on the faces of the simplicial complex.
        incidence_2 : torch.sparse
            shape=[n_edges, n_faces]
            Incidence matrix mapping faces to edges (B_2).

        Returns
        -------
        x_2 : torch.tensor
            shape=[n_faces, out_channels]
            Output features on the faces of the simplicial complex.
        """
        incidence_2_transpose = incidence_2.to_dense().T.to_sparse()
        if x_2.shape[-2] != incidence_2.shape[-1]:
            raise ValueError(
                f"Shape of input face features does not have the correct number of faces {incidence_2.shape[-1]}."
            )
        x_1 = self.conv_level1_2_to_1(x_2, incidence_2)
        x_2 = self.conv_level2_1_to_2(x_1, incidence_2_transpose)
        return x_2
