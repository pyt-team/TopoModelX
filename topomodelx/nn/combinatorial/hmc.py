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
        each input signal (nodes, edges, and faces) for the k-th layer .
    num_classes : int
        Number of classes.
    negative_slope : float
        Negative slope for the LeakyReLU activation.
    """

    def __init__(
        self,
        channels_per_layer,
        num_classes,
        negative_slope=0.2,
        update_func_attention="relu",
        update_func_aggregation="relu",
    ) -> None:
        def check_channels_consistency():
            assert len(channels_per_layer) > 0
            for i in range(len(channels_per_layer) - 1):
                assert channels_per_layer[i][2][0] == channels_per_layer[i + 1][0][0]
                assert channels_per_layer[i][2][1] == channels_per_layer[i + 1][0][1]
                assert channels_per_layer[i][2][2] == channels_per_layer[i + 1][0][2]

        super().__init__()
        self.num_classes = num_classes
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

        self.l0 = torch.nn.Linear(channels_per_layer[-1][2][0], num_classes)
        self.l1 = torch.nn.Linear(channels_per_layer[-1][2][1], num_classes)
        self.l2 = torch.nn.Linear(channels_per_layer[-1][2][2], num_classes)

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
    ) -> torch.Tensor:
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
        y_hat : torch.Tensor, shape=[num_classes]
            Vector embedding that represents the probability of the input mesh to belong to each class.
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

        x_0 = self.l0(x_0)
        x_1 = self.l1(x_1)
        x_2 = self.l2(x_2)

        # Sum all the elements in the dimension zero
        x_0 = torch.nanmean(x_0, dim=0)
        x_1 = torch.nanmean(x_1, dim=0)
        x_2 = torch.nanmean(x_2, dim=0)

        return x_0 + x_1 + x_2
