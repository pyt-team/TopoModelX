import torch
import torch.nn.functional as F

from topomodelx.nn.conv import MessagePassingConv


class TemplateLayer(torch.nn.Module):
    """Template Layer with two message passing steps.
    We show how to

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    intra_aggr : string
        Aggregation method.
        (Inter-neighborhood).
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
        incidence_matrix_2,
    ):
        super().__init__()
        self.incidence_matrix_2 = incidence_matrix_2

        incidence_matrix_2_transpose = incidence_matrix_2.to_dense().T.to_sparse()

        self.message_level1_2_to_1 = MessagePassingConv(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            neighborhood=incidence_matrix_2,
            inter_agg_norm=True,
            update_on_message="sigmoid",
        )
        self.message_level2_1_to_2 = MessagePassingConv(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            neighborhood=incidence_matrix_2_transpose,
            inter_agg_norm=True,
            update_on_message="sigmoid",
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.message_level1_2_to_1.reset_parameters()
        self.message_level2_1_to_2.reset_parameters()

    def forward(self, x):
        r"""Forward computation.

        Parameters
        ----------
        x: torch.tensor, shape=[n_faces, in_channels]
            Input features on the faces of the simplicial complex.
        """
        if x.shape[-2] != self.incidence_matrix_2.shape[-1]:
            raise ValueError(
                f"Shape of input face features does not have the correct number of faces {self.incidence_matrix_2.shape[-1]}."
            )
        x_edges = self.message_level1_2_to_1(x)
        x = self.message_level2_1_to_2(x_edges)
        return x
