"""CWN class."""

import torch
import torch.nn.functional as F

from topomodelx.nn.cell.cwn_layer import CWNLayer


class CWN(torch.nn.Module):
    """Implementation of a specific version of CW network [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    hid_channels : int
        Dimension of hidden features.
    num_classes : int
        Number of classes.
    n_layers : int
        Number of CWN layers.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels,
        num_classes,
        n_layers,
    ):
        super().__init__()
        self.proj_0 = torch.nn.Linear(in_channels_0, hid_channels)
        self.proj_1 = torch.nn.Linear(in_channels_1, hid_channels)
        self.proj_2 = torch.nn.Linear(in_channels_2, hid_channels)

        layers = []
        for _ in range(n_layers):
            layers.append(
                CWNLayer(
                    in_channels_0=hid_channels,
                    in_channels_1=hid_channels,
                    in_channels_2=hid_channels,
                    out_channels=hid_channels,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

        self.lin_0 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_1 = torch.nn.Linear(hid_channels, num_classes)
        self.lin_2 = torch.nn.Linear(hid_channels, num_classes)

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        neighborhood_1_to_1,
        neighborhood_2_to_1,
        neighborhood_0_to_1,
    ):
        """Forward computation through projection, convolutions, linear layers and average pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = [n_nodes, in_channels_0]
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = [n_edges, in_channels_1]
            Input features on the edges (1-cells).
        x_2 : torch.Tensor, shape = [n_faces, in_channels_2]
            Input features on the faces (2-cells).
        neighborhood_1_to_1 : tensor, shape = [n_edges, n_edges]
            Upper-adjacency matrix of rank 1.
        neighborhood_2_to_1 : tensor, shape = [n_edges, n_faces]
            Boundary matrix of rank 2.
        neighborhood_0_to_1 : tensor, shape = [n_edges, n_nodes]
            Coboundary matrix of rank 1.

        Returns
        -------
        _ : tensor, shape = [1]
            Label assigned to whole complex.
        """
        x_0 = F.elu(self.proj_0(x_0))
        x_1 = F.elu(self.proj_1(x_1))
        x_2 = F.elu(self.proj_2(x_2))

        for layer in self.layers:
            x_1 = layer(
                x_0,
                x_1,
                x_2,
                neighborhood_1_to_1,
                neighborhood_2_to_1,
                neighborhood_0_to_1,
            )

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
