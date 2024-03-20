"""SCCNN implementation for complex classification."""
import torch

from topomodelx.nn.simplicial.sccnn_layer import SCCNNLayer


class SCCNN(torch.nn.Module):
    """SCCNN implementation for complex classification.

    Note: In this task, we can consider the output on any order of simplices for the
    classification task, which of course can be amended by a readout layer.

    Parameters
    ----------
    in_channels_all: tuple of int
        Dimension of input features on (nodes, edges, faces).
    hidden_channels_all: tuple of int
        Dimension of features of hidden layers on (nodes, edges, faces).
    conv_order: int
        Order of convolutions, we consider the same order for all convolutions.
    sc_order: int
        Order of simplicial complex.
    aggr_norm: bool
        Whether to normalize the aggregation.
    update_func: str
        Update function for the simplicial complex convolution.
    n_layers: int
        Number of layers.

    """

    def __init__(
        self,
        in_channels_all,
        hidden_channels_all,
        conv_order,
        sc_order,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # first layer
        # we use an MLP to map the features on simplices of different dimensions to the same dimension
        self.in_linear_0 = torch.nn.Linear(in_channels_all[0], hidden_channels_all[0])
        self.in_linear_1 = torch.nn.Linear(in_channels_all[1], hidden_channels_all[1])
        self.in_linear_2 = torch.nn.Linear(in_channels_all[2], hidden_channels_all[2])

        self.layers = torch.nn.ModuleList(
            SCCNNLayer(
                in_channels=hidden_channels_all,
                out_channels=hidden_channels_all,
                conv_order=conv_order,
                sc_order=sc_order,
                aggr_norm=aggr_norm,
                update_func=update_func,
            )
            for _ in range(n_layers)
        )

    def forward(self, x_all, laplacian_all, incidence_all):
        """Forward computation.

        Parameters
        ----------
        x_all : tuple of tensors
            Tuple of feature tensors (node, edge, face).
            Each entry shape = (n_simplices, channels).

        laplacian_all : tuple of tensors
            Tuple of Laplacian tensors (graph laplacian L0, down edge laplacian L1_d, upper edge laplacian L1_u, face laplacian L2).
            Each entry shape = (n_simplices,n_simplices).

        incidence_all : tuple of tensors
            Tuple of order 1 and 2 incidence matrices.
            Shape of B1 = [n_nodes, n_edges].
            Shape of B2 = [n_edges, n_faces].

        Returns
        -------
        x_all : tuple of tensors
            Tuple of final hidden state tensors (node, edge, face).
            Each entry shape = (n_simplices, channels).
        """
        x_0, x_1, x_2 = x_all
        in_x_0 = self.in_linear_0(x_0)
        in_x_1 = self.in_linear_1(x_1)
        in_x_2 = self.in_linear_2(x_2)

        # Forward through SCCNN
        x_all = (in_x_0, in_x_1, in_x_2)
        for layer in self.layers:
            x_all = layer(x_all, laplacian_all, incidence_all)

        return x_all
