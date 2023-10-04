"""CAN class."""

import torch
import torch.nn.functional as F

from topomodelx.nn.cell.can_layer import CANLayer, MultiHeadLiftLayer, PoolLayer


class CAN(torch.nn.Module):
    """CAN (Cell Attention Network) [1]_ module for graph classification.

    Parameters
    ----------
    in_channels_0: int
        Number of input channels for the node-level input.
    in_channels_1: int
        Number of input channels for the edge-level input.
    out_channels: int
        Number of output channels.
    num_classest: int
        Number of output classes.
    dropout: float, optional
        Dropout probability. Default is 0.5.
    heads: int, optional
        Number of attention heads. Default is 3.
    concat: bool, optional
        Whether to concatenate the output channels of attention heads. Default is True.
    skip_connection: bool, optional
        Whether to use skip connections. Default is True.
    att_activation: torch.nn.Module, optional
        Activation function for attention mechanism. Default is torch.nn.LeakyReLU(0.2).
    n_layers: int, optional
        Number of CAN layers. Default is 2.
    att_lift: bool, optional
        Whether to apply a lift the signal from node-level to edge-level input. Default is True.

    References
    ----------
    .. [1] Giusti, Battiloro, Testa, Di Lorenzo, Sardellitti and Barbarossa.
        Cell attention networks (2022).
        Paper: https://arxiv.org/pdf/2209.08179.pdf
        Repository: https://github.com/lrnzgiusti/can
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        out_channels,
        num_classes,
        dropout=0.5,
        heads=3,
        concat=True,
        skip_connection=True,
        att_activation=torch.nn.LeakyReLU(0.2),
        n_layers=2,
        att_lift=True,
    ):
        super().__init__()

        if att_lift:
            self.lift_layer = MultiHeadLiftLayer(
                in_channels_0=in_channels_0,
                heads=in_channels_0,
                signal_lift_dropout=0.5,
            )
            in_channels_1 = in_channels_1 + in_channels_0

        layers = []

        layers.append(
            CANLayer(
                in_channels=in_channels_1,
                out_channels=out_channels,
                heads=heads,
                concat=concat,
                skip_connection=skip_connection,
                att_activation=att_activation,
                aggr_func="sum",
                update_func="relu",
            )
        )

        for _ in range(n_layers - 1):
            layers.append(
                CANLayer(
                    in_channels=out_channels * heads,
                    out_channels=out_channels,
                    dropout=dropout,
                    heads=heads,
                    concat=concat,
                    skip_connection=skip_connection,
                    att_activation=att_activation,
                    aggr_func="sum",
                    update_func="relu",
                )
            )

            layers.append(
                PoolLayer(
                    k_pool=0.5,
                    in_channels_0=out_channels * heads,
                    signal_pool_activation=torch.nn.Sigmoid(),
                    readout=True,
                )
            )

        self.layers = torch.nn.ModuleList(layers)
        self.lin_0 = torch.nn.Linear(heads * out_channels, 128)
        self.lin_1 = torch.nn.Linear(128, num_classes)

    def forward(
        self, x_0, x_1, neighborhood_0_to_0, lower_neighborhood, upper_neighborhood
    ):
        """Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        lower_neighborhood : torch.Tensor, shape = (-, -)
            Lower Neighbourhood matrix.
        upper_neighborhood : torch.Tensor, shape = (-, -)
            Upper neighbourhood matrix.

        Returns
        -------
        torch.Tensor
        """
        if hasattr(self, "lift_layer"):
            x_1 = self.lift_layer(x_0, neighborhood_0_to_0, x_1)

        for layer in self.layers:
            if isinstance(layer, PoolLayer):
                x_1, lower_neighborhood, upper_neighborhood = layer(
                    x_1, lower_neighborhood, upper_neighborhood
                )
            else:
                x_1 = layer(x_1, lower_neighborhood, upper_neighborhood)
                x_1 = F.dropout(x_1, p=0.5, training=self.training)

        # max pooling over all nodes in each graph
        x = x_1.max(dim=0)[0]

        # Feed-Foward Neural Network to predict the graph label
        out = self.lin_1(torch.nn.functional.relu(self.lin_0(x)))

        return out
