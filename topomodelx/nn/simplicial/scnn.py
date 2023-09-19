"""Simplicial convolutional neural network implementation for complex classification."""
import torch

from topomodelx.nn.simplicial.scnn_layer import SCNNLayer


class SCNN(torch.nn.Module):
    """Simplicial convolutional neural network implementation for complex classification.

    Note: At the last layer, we obtain the output on simplcies, e.g., edges.
    To perform the complex classification task for this challenge, we consider pass the final output to a linear layer and compute the average.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    intermediate_channels : int
        Dimension of features of intermediate layers.
    out_channels : int
        Dimension of output features.
    conv_order_down : int
        Order of lower convolution.
    conv_order_up : int
        Order of upper convolution.
    aggr : bool
        Whether to aggregate features on the nodes into 1 feature for the whole complex.
        Default: False.
    n_layers : int
        Number of layers.
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        aggr=False,
        aggr_norm=False,
        update_func=None,
        n_layers=2,
    ):
        super().__init__()
        # First layer -- initial layer has the in_channels as input, and inter_channels as the output
        layers = [
            SCNNLayer(
                in_channels=in_channels,
                out_channels=intermediate_channels,
                conv_order_down=conv_order_down,
                conv_order_up=conv_order_up,
            )
        ]

        for _ in range(n_layers - 1):
            layers.append(
                SCNNLayer(
                    in_channels=intermediate_channels,
                    out_channels=out_channels,
                    conv_order_down=conv_order_down,
                    conv_order_up=conv_order_up,
                    aggr_norm=aggr_norm,
                    update_func=update_func,
                )
            )
        self.aggr = aggr
        self.linear = torch.nn.Linear(out_channels, 1)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, laplacian_down, laplacian_up):
        """Forward computation.

        Parameters
        ----------
        x : tensor, shape=[n_simplices, channels]
            node/edge/face features
        laplacian_down : tensor, shape=[n_simplices, n_simplices]
            Down Laplacian.
            For node features, laplacian_down = None
        laplacian_up: tensor, shape=[n_edges, n_nodes]
            Up Laplacian.

        Returns
        -------
        one_dimensional_cells_mean : tensor, shape=[n_simplices, 1]
            Mean on one-dimensional cells.
        """
        for layer in self.layers:
            x = layer(x, laplacian_down, laplacian_up)

        x_1 = self.linear(x)
        if not self.aggr:
            return x_1
        one_dimensional_cells_mean = torch.nanmean(x_1, dim=0)
        one_dimensional_cells_mean[torch.isnan(one_dimensional_cells_mean)] = 0

        return one_dimensional_cells_mean
