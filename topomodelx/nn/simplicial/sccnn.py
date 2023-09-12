"""SCCNN implementation for complex classification."""
import torch

from topomodelx.nn.simplicial.sccnn_layer import SCCNNLayer


class SCCNN(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all: tuple of int
        Dimension of input features on (nodes, edges, faces)
    intermediate_channels_all: tuple of int
        Dimension of features of intermediate layers on (nodes, edges, faces)
    out_channels_all: tuple of int
        Dimension of output features on (nodes, edges, faces)
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions
    sc_order: int
        SC order
    n_layers: int
        Numer of layers
    """

    def __init__(
        self,
        in_channels_all,
        intermediate_channels_all,
        out_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        num_classes=1,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(
            in_channels_all[0], intermediate_channels_all[0]
        )
        self.in_linear_1 = torch.nn.Linear(
            in_channels_all[1], intermediate_channels_all[1]
        )
        self.in_linear_2 = torch.nn.Linear(
            in_channels_all[2], intermediate_channels_all[2]
        )

        layers = []
        for _ in range(n_layers):
            layers.append(
                SCCNNLayer(
                    in_channels=intermediate_channels_all,
                    out_channels=out_channels_all,
                    conv_order=conv_order,
                    sc_order=sc_order,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

        out_channels_0, out_channels_1, out_channels_2 = out_channels_all
        self.out_linear_0 = torch.nn.Linear(out_channels_0, num_classes)
        self.out_linear_1 = torch.nn.Linear(out_channels_1, num_classes)
        self.out_linear_2 = torch.nn.Linear(out_channels_2, num_classes)

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x: tuple tensors
            (node, edge, face) features
            each entry shape = [n_simplices, channels]

        laplacian: tuple of tensors
            (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2)
            each entry shape = [n_simplices,n_simplices]

        incidence_1: tuple of tensors
            tuple pf order 1 and 2 incidence matrices
            shape of B1 = [n_nodes, n_edges]
            shape of B2 = [n_edges, n_faces]

        Returns
        -------
        _ : tensor, shape = [1]
            Label assigned to whole complex.
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        """
        We pass the output on the nodes, edges and triangles to a linear layer and use the sum of the averages of outputs on each simplex for labels of complex
        """
        x_0, x_1, x_2 = x_all

        x_0 = self.out_linear_0(x_0)
        x_1 = self.out_linear_1(x_1)
        x_2 = self.out_linear_2(x_2)

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


class SCCNNComplex(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all: tuple of int
        Dimension of input features on (nodes, edges, faces).
    intermediate_channels_all: tuple of int
        Dimension of features of intermediate layers on (nodes, edges, faces).
    out_channels_all: tuple of int
        Dimension of output features on (nodes, edges, faces)
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order: int
        Order of simplicial complex.
    n_layers: int
        Number of layers.

    """

    def __init__(
        self,
        in_channels_all,
        intermediate_channels_all,
        out_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(
            in_channels_all[0], intermediate_channels_all[0]
        )
        self.in_linear_1 = torch.nn.Linear(
            in_channels_all[1], intermediate_channels_all[1]
        )
        self.in_linear_2 = torch.nn.Linear(
            in_channels_all[2], intermediate_channels_all[2]
        )

        layers = []
        for _ in range(n_layers):
            layers.append(
                SCCNNLayer(
                    in_channels=intermediate_channels_all,
                    out_channels=out_channels_all,
                    conv_order=conv_order,
                    sc_order=sc_order,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )

        self.layers = torch.nn.ModuleList(layers)

        out_channels_0, out_channels_1, out_channels_2 = out_channels_all
        self.out_linear_0 = torch.nn.Linear(out_channels_0, 2)

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x: tuple tensors
            (node, edge, face) features
            each entry shape = [n_simplices, channels]

        laplacian: tuple of tensors
            (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2)
            each entry shape = [n_simplices,n_simplices]

        incidence_1: tuple of tensors
            tuple pf order 1 and 2 incidence matrices
            shape of B1 = [n_nodes, n_edges]
            shape of B2 = [n_edges, n_faces]

        Returns
        -------
        _ : tensor
            shape = [n_nodes, 2]
            One-hot labels assigned to nodes.
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        """
        We pass the output on the nodes to a linear layer and use that to generate a probability label for nodes
        """
        x_0, _, _ = x_all
        logits = self.out_linear_0(x_0)
        label = torch.sigmoid(logits)

        return label
