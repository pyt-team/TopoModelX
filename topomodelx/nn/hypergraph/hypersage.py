"""HyperSAGE Layer."""

import torch

from topomodelx.nn.hypergraph.hypersage_layer import HyperSAGELayer


class HyperSAGE(torch.nn.Module):
    """Neural network implementation of HyperSAGE [1]_ for hypergraph classification.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    n_layer : int, default = 2
        Amount of message passing layers.

    References
    ----------
    .. [1] Arya, Gupta, Rudinac and Worring.
        HyperSAGE: Generalizing inductive representation learning on hypergraphs (2020).
        https://arxiv.org/abs/2010.04558
    """

    def __init__(
            self, 
            in_channels, 
            hidden_channels, 
            n_layers=2, 
            **kwargs
        ):
        super().__init__()
        layers = []
        layers.append(
            HyperSAGELayer(in_channels=in_channels, out_channels=hidden_channels, **kwargs)
        )
        for _ in range(1, n_layers):
            layers.append(
                HyperSAGELayer(
                    in_channels=hidden_channels, 
                    out_channels=hidden_channels, 
                    **kwargs
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        
    def forward(self, x_0, incidence):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_nodes, features_nodes)
            Edge features.
        incidence : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0 = layer.forward(x_0, incidence)
        
        return x_0