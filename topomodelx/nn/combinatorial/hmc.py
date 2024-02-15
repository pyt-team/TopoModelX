"""HOAN mesh classifer class."""


import torch

from topomodelx.nn.combinatorial.hmc_layer import HMCLayer


class HMC(torch.nn.Module):
    """Higher Order Attention Network for Mesh Classification.

    Parameters
    ----------
    channels_per_layer : list of list of list of int
        Number of input, intermediate, and output channels for each Higher Order Attention Layer.
        The length of the list corresponds to the number of layers. Each element k of the list
        is a list consisting of other 3 lists. The first list contains the number of input channels
        for each input signal (nodes, edges, and faces) for the k-th layer. The second list
        contains the number of intermediate channels for each input signal (nodes, edges, and
        faces) for the k-th layer. Finally, the third list contains the number of output channels for
        each input signal (nodes, edges, and faces) for the k-th layer.
    negative_slope : float
        Negative slope for the LeakyReLU activation.
    update_func_attention : str
        Update function for the attention mechanism. Default is "relu".
    update_func_aggregation : str
        Update function for the aggregation mechanism. Default is "relu".
    """

    def __init__(
        self,
        channels_per_layer,
        negative_slope=0.2,
        update_func_attention="relu",
        update_func_aggregation="relu",
    ):
        def check_channels_consistency():
            """Check that the number of input, intermediate, and output channels is consistent."""
            assert len(channels_per_layer) > 0
            for i in range(len(channels_per_layer) - 1):
                assert channels_per_layer[i][2][0] == channels_per_layer[i + 1][0][0]
                assert channels_per_layer[i][2][1] == channels_per_layer[i + 1][0][1]
                assert channels_per_layer[i][2][2] == channels_per_layer[i + 1][0][2]

        super().__init__()
        check_channels_consistency()
        self.layers = torch.nn.ModuleList(
            [
                HMCLayer(
                    in_channels=in_channels,
                    intermediate_channels=intermediate_channels,
                    out_channels=out_channels,
                    negative_slope=negative_slope,
                    softmax_attention=True,
                    update_func_attention=update_func_attention,
                    update_func_aggregation=update_func_aggregation,
                )
                for in_channels, intermediate_channels, out_channels in channels_per_layer
            ]
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        neighborhood_0_to_0,
        neighborhood_1_to_1,
        neighborhood_2_to_2,
        neighborhood_0_to_1,
        neighborhood_1_to_2,
    ):
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor
            Input features on nodes.
        x_1 : torch.Tensor
            Input features on edges.
        x_2 : torch.Tensor
            Input features on faces.
        neighborhood_0_to_0 : torch.Tensor
            Adjacency  matrix from nodes to nodes.
        neighborhood_1_to_1 : torch.Tensor
            Adjacency  matrix from edges to edges.
        neighborhood_2_to_2 : torch.Tensor
            Adjacency  matrix from faces to faces.
        neighborhood_0_to_1 : torch.Tensor
            Incidence matrix from nodes to edges.
        neighborhood_1_to_2 : torch.Tensor
            Incidence matrix from edges to faces.

        Returns
        -------
        torch.Tensor, shape = (n_nodes, out_channels_0)
            Final hidden states of the nodes (0-cells).
        torch.Tensor, shape = (n_edges, out_channels_1)
            Final hidden states the edges (1-cells).
        torch.Tensor, shape = (n_faces, out_channels_2)
            Final hidden states of the faces (2-cells).
        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(
                x_0,
                x_1,
                x_2,
                neighborhood_0_to_0,
                neighborhood_1_to_1,
                neighborhood_2_to_2,
                neighborhood_0_to_1,
                neighborhood_1_to_2,
            )

        return x_0, x_1, x_2
