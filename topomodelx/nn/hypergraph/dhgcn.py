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
    n_layer : int, default = 2
        Amount of message passing layers.
    task_level: str, default="graph"
        Level of the task. Either "graph" or "node".
        If "graph", the output is pooled over all nodes in the hypergraph.

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
        out_channels, 
        n_layers=1, 
        task_level="graph"
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
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = True if task_level == "graph" else False
        

    def forward(self, x_0):
        """Forward computation through layers, then global average pooling, then linear layer.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, node_channels)
            Edge features.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0 = layer(x_0)
        
        # Pool over all nodes in the hypergraph 
        if self.out_pool is True:
            x = torch.max(x_0, dim=0)[0]
        else:
            x = x_0        
        
        return self.linear(x)
