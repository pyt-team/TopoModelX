import torch
import torch.nn.functional as F

from topomodelx.nn.conv import MessagePassingConv


class TemplateLayer(torch.nn.Module):
    """Template Layer.

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
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.out_channels = out_channels
        self.incidence_matrix_2 = incidence_matrix_2
        self.incidence_matrix_2_transpose = incidence_matrix_2.to_dense().T.to_sparse()
        self.initialization = initialization

        self.level1 = MessagePassingConv(
            in_channels, intermediate_channels, update="sigmoid"
        )
        self.level2 = MessagePassingConv(
            intermediate_channels, out_channels, update="sigmoid"
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.level1.reset_parameters()
        self.level2.reset_parameters()

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

        x = self.level1(x, self.incidence_matrix_2)

        x = self.level2(x, self.incidence_matrix_2_transpose)

        return x
