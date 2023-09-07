"""UniGCNII class."""

import torch

from topomodelx.nn.hypergraph.unigin_layer import UniGINLayer


class UniGIN(torch.nn.Module):
    """Neural network implementation of UniGIN for hypergraph classification.

    Parameters
    ----------
    in_channels_node : int
        Dimension of node features
    n_layer : 2
        Amount of message passing layers.
    """

    def __init__(
        self, in_channels_node, intermediate_channels, out_channels, n_layers=2
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            mlp = torch.nn.Sequential(
                torch.nn.Linear(intermediate_channels, 2 * intermediate_channels),
                torch.nn.ReLU(),
                torch.nn.Linear(2 * intermediate_channels, intermediate_channels),
            )
            layers.append(
                UniGINLayer(
                    nn=mlp,
                    in_channels=intermediate_channels,
                )
            )

        self.inp_embed = torch.nn.Linear(in_channels_node, intermediate_channels)
        self.layers = torch.nn.ModuleList(layers)
        self.out_decoder = torch.nn.Linear(intermediate_channels, out_channels)

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_0 : tensor
            shape = [n_nodes, in_channels_node]
            Edge features.

        incidence_1 : tensor
            shape = [n_nodes, n_edges]
            Boundary matrix of rank 1.

        Returns
        -------
        _ : tensor
            shape = [1]
            Label assigned to whole complex.
        """
        x_0 = self.inp_embed(x_0)
        for layer in self.layers:
            x_0 = layer(x_0, incidence_1)
        pooled_x_0 = torch.mean(x_0, dim=0)
        return torch.sigmoid(self.out_decoder(pooled_x_0))
