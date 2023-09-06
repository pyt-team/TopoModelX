"""SCA with AMPS."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.nn.simplicial.sca_cmps_layer import SCACMPSLayer


class AMPSSCA(torch.nn.Module):
    """SCA with AMPS.

    Parameters
    ----------
    channels_list: list[int]
        Dimension of features on each node, edge, simplex, tetahedron,... respectively
    complex_dimension: int
        Highest dimension of simplicial complex feature being trained on.
    num_classes : int
        Dimension to which the complex embeddings will be projected.
    att : bool
        Whether to use attention.
    """

    def __init__(
        self,
        channels_list,
        complex_dim,
        num_classes,
        n_layers=2,
        att=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.channels_list = channels_list
        self.num_classes = num_classes

        layers = []
        for i in range(n_layers):
            layers.append(SCACMPSLayer(channels_list, complex_dim, att))

        self.layers = torch.nn.ModuleList(layers)
        self.lin0 = torch.nn.Linear(channels_list[0], num_classes)
        self.lin1 = torch.nn.Linear(channels_list[1], num_classes)
        self.lin2 = torch.nn.Linear(channels_list[2], num_classes)
        self.aggr = Aggregation(
            aggr_func="mean",
            update_func="sigmoid",
        )

    def forward(self, x_list, down_lap_list, incidencet_list):
        """Forward computation through layers, then linear layers, then avg pooling.

        Parameters
        ----------
        x_list: list[torch.Tensor]
            List of tensor inputs for each dimension of the complex (nodes, edges, etc.).
        down_lap_list: list[torch.Tensor]
            List of the down laplacian matrix for each dimension in the complex starting at edges.
        incidencet_list: list[torch.Tensor]
            List of the transpose incidence matrices for the edges and faces.

        Returns
        -------
        _ : tensor, shape = [1]
            Label assigned to whole complex.
        """
        for i in range(self.n_layers):
            x_list = self.layers[i](x_list, down_lap_list, incidencet_list)

        x_0 = self.lin0(x_list[0])
        x_1 = self.lin1(x_list[1])
        x_2 = self.lin2(x_list[2])

        two_dimensional_cells_mean = torch.nanmean(x_2, dim=0)
        two_dimensional_cells_mean[torch.isnan(two_dimensional_cells_mean)] = 0
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0
        zero_dimensional_cells_mean = torch.nanmean(x_0, dim=0)
        zero_dimensional_cells_mean[torch.isnan(zero_dimensional_cells_mean)] = 0

        x_2f = torch.flatten(two_dimensional_cells_mean)
        x_1f = torch.flatten(one_dimensional_cells_mean)
        x_0f = torch.flatten(zero_dimensional_cells_mean)

        return x_0f + x_1f + x_2f