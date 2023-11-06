"""Simplex Convolutional Network Implementation for binary node classification."""
import torch

from topomodelx.nn.simplicial.scn2_layer import SCN2Layer


class SCN2(torch.nn.Module):
    """Simplex Convolutional Network Implementation for binary node classification.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes.
    in_channels_1 : int
        Dimension of input features on edges.
    in_channels_2 : int
        Dimension of input features on faces.
    n_layers : int
        Amount of message passing layers.

    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                SCN2Layer(
                    in_channels_0=in_channels_0,
                    in_channels_1=in_channels_1,
                    in_channels_2=in_channels_2,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
            Node features.

        x_1 : torch.Tensor, shape = (n_edges, channels)
            Edge features.

        x_2 : torch.Tensor, shape = (n_faces, channels)
            Face features.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
            Final node hidden states.

        x_1 : torch.Tensor, shape = (n_nodes, channels)
            Final edge hidden states.

        x_2 : torch.Tensor, shape = (n_nodes, channels)
            Final face hidden states.

        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)

        return x_0, x_1, x_2
