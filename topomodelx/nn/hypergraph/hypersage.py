"""HyperSAGE Layer."""

import torch

from topomodelx.nn.hypergraph.hypersage_layer import HyperSAGELayer


class HyperSAGE(torch.nn.Module):
    """Neural network implementation of HyperSAGE for hypergraph classification.

    Parameters
    ----------
    channels_edge : int
        Dimension of edge features
    channels_node : int
        Dimension of node features
    n_layer : int
        Amount of message passing layers. Default is 2.

    References
    ----------
    .. [AGRW20] Devanshu Arya, Deepak K Gupta, Stevan Rudinac and Marcel Worring.
        HyperSAGE: Generalizing inductive representation learning on hypergraphs.
        arXiv preprint arXiv:2010.04558. 2020
    """

    def __init__(self, in_channels, out_channels, n_layers=2, **kwargs):
        super().__init__()
        layers = []
        layers.append(
            HyperSAGELayer(in_channels=in_channels, out_channels=out_channels, **kwargs)
        )
        for _ in range(1, n_layers):
            layers.append(
                HyperSAGELayer(
                    in_channels=out_channels, out_channels=out_channels, **kwargs
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(out_channels, 1)

    def forward(self, x, incidence):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x: tensor
            shape = [n_nodes, features_nodes]
            Edge features.

        incidence: tensor
            shape = [n_nodes, n_edges]
            Boundary matrix of rank 1.

        Returns
        -------
        _ : tensor
            shape = [1]
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x = layer.forward(x, incidence)
        pooled_x = torch.max(x, dim=0)[0]
        return torch.sigmoid(self.linear(pooled_x))[0]
