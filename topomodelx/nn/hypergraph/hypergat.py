"""HyperGat Layer."""

import torch

from topomodelx.nn.hypergraph.hypergat_layer import HyperGATLayer


class HyperGAT(torch.nn.Module):
    """Neural network implementation of Template for hypergraph classification [1]_.

    Parameters
    ----------
    in_channels : int
        Dimension of the input features.
    hidden_channels : int
        Dimension of the hidden features.
    out_channels : int
        Dimension of the output features.
    n_layer : int, default = 2
        Amount of message passing layers.
    task_level: str, default="graph"
        Level of the task. Either "graph" or "node".
        If "graph", the output is pooled over all nodes in the hypergraph.


    References
    ----------
    .. [1] Ding, Wang, Li, Li and Huan Liu.
        EMNLP, 2020.
        https://aclanthology.org/2020.emnlp-main.399.pdf
    """

    def __init__(
            self, 
            in_channels,
            hidden_channels,
            out_channels, 
            n_layers=2,
            task_level="graph",
        ):
        super().__init__()
        layers = []
        layers.append(HyperGATLayer(in_channels=in_channels, out_channels=hidden_channels))
        for _ in range(1, n_layers):
            layers.append(
                HyperGATLayer(in_channels=hidden_channels, out_channels=hidden_channels)
            )
        self.layers = torch.nn.ModuleList(layers)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = True if task_level == "graph" else False

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x_1 : torch.Tensor
            shape = (n_edges, channels_edge)
            Edge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for layer in self.layers:
            x_0 = layer.forward(x_0, incidence_1)
        
        # Pool over all nodes in the hypergraph 
        if self.out_pool is True:
            x = torch.max(x_0, dim=0)[0]
        else:
            x = x_0

        return self.linear(x)
