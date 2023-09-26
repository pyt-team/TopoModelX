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
    num_classes : int
        Number of classes.
    n_layers : int
        Amount of message passing layers.

    """

    def __init__(
        self, in_channels_0, in_channels_1, in_channels_2, num_classes, n_layers=2
    ):
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
        self.lin_0 = torch.nn.Linear(in_channels_0, num_classes)
        self.lin_1 = torch.nn.Linear(in_channels_1, num_classes)
        self.lin_2 = torch.nn.Linear(in_channels_2, num_classes)

    def forward(self, x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2):
        """Forward computation.

        Parameters
        ----------
        x_0 : tensor, shape = (n_nodes, channels)
            Node features.

        Returns
        -------
        _ : tensor, shape = (n_nodes, 2)
            One-hot labels assigned to nodes.

        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2)
        x_0 = self.lin_0(x_0)
        x_1 = self.lin_1(x_1)
        x_2 = self.lin_2(x_2)

        # Take the average of the 2D, 1D, and 0D cell features. If they are NaN, convert them to 0.
        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0
        # Return the sum of the averages
        return (
            two_dimensional_cells_mean
            + one_dimensional_cells_mean
            + zero_dimensional_cells_mean
        )
