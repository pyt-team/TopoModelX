"""DHGCN class."""

import torch

from topomodelx.nn.hypergraph.dhgcn_layer import DHGCNLayer


class DHGCN(torch.nn.Module):
    """Neural network implementation of DHGCN [1]_ for hypergraph classification.

    Only dynamic topology is used here.

    Parameters
    ----------
    channels_edge : int
        Dimension of edge features
    channels_node : int
        Dimension of node features
    n_layer : 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Yin, Feng, Luo, Zhang, Wang, Luo, Chen and Hua.
        Dynamic hypergraph convolutional network (2022).
        https://ieeexplore.ieee.org/abstract/document/9835240
    """

    def __init__(self, channels_node, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                DHGCNLayer(
                    in_channels=channels_node,
                    intermediate_channels=channels_node,
                    out_channels=channels_node,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(channels_node, 1)

    def forward(self, x_0):
        """Forward computation through layers, then global average pooling, then linear layer.

        Parameters
        ----------
        x_0 : tensor
            shape = [n_nodes, node_channels]
            Edge features.

        Returns
        -------
        _ : tensor
            shape = [1]
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0 = layer(x_0)
        pooled_x = torch.mean(x_0, dim=0)
        output = self.linear(pooled_x)
        return output[0]
