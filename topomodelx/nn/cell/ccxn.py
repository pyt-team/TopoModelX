"""CCXN class."""

import torch

from topomodelx.nn.cell.ccxn_layer import CCXNLayer


class CCXN(torch.nn.Module):
    """CCXN [1]_.

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
        Number of CCXN layers.
    att : bool
        Whether to use attention.

    References
    ----------
    .. [1] Hajij, Istvan, Zamzmi.
        Cell complex neural networks.
        Topological data analysis and beyond workshop at NeurIPS 2020.
        https://arxiv.org/pdf/2010.00743.pdf
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        num_classes,
        n_layers=2,
        att=False,
    ):
        super().__init__()

        self.layers = torch.nn.ModuleList(
            CCXNLayer(
                in_channels_0=in_channels_0,
                in_channels_1=in_channels_1,
                in_channels_2=in_channels_2,
                att=att,
            )
            for _ in range(n_layers)
        )
        self.lin_0 = torch.nn.Linear(in_channels_0, num_classes)
        self.lin_1 = torch.nn.Linear(in_channels_1, num_classes)
        self.lin_2 = torch.nn.Linear(in_channels_2, num_classes)

    def forward(self, x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2):
        """Forward computation through layers, then linear layers, then avg pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        neighborhood_0_to_0 : torch.Tensor, shape = (n_nodes, n_nodes)
            Adjacency matrix of rank 0 (up).
        neighborhood_1_to_2 : torch.Tensor, shape = (n_faces, n_edges)
            Transpose of boundary matrix of rank 2.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2)
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
