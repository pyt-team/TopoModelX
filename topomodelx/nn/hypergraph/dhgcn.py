"""DHGCN class."""

import torch

from topomodelx.nn.hypergraph.dhgcn_layer import DHGCNLayer


class DHGCN(torch.nn.Module):
    """Neural network implementation of DHGCN [1]_ for hypergraph classification.

    Only dynamic topology is used here.

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
    .. [1] Yin, Feng, Luo, Zhang, Wang, Luo, Chen and Hua.
        Dynamic hypergraph convolutional network (2022).
        https://ieeexplore.ieee.org/abstract/document/9835240
    """

    def __init__(
        self, 
        in_channels, 
        hidden_channels,  
        n_layers=1, 
    ):
        super().__init__()
        layers = []
        layers.append(
            DHGCNLayer(
                in_channels=in_channels,
                intermediate_channels=hidden_channels,
                out_channels=hidden_channels,
            )
        )
        for _ in range(n_layers - 1):
            layers.append(
                DHGCNLayer(
                    in_channels=hidden_channels,
                    intermediate_channels=hidden_channels,
                    out_channels=hidden_channels,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
       
        

    def forward(self, x_0):
        """Forward computation through layers, then global average pooling, then linear layer.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, node_channels)
            Edge features.

        Returns
        -------
        x_0 : torch.Tensor
            Output node features.
        x_1 : torch.Tensor
            Output hyperedge features.
        """
        for layer in self.layers:
            x_0, x_1 = layer(x_0)
        
        return (x_0, x_1)
