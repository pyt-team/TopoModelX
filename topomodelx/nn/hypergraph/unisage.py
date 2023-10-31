"""UniSAGE class."""

import torch

from topomodelx.nn.hypergraph.unisage_layer import UniSAGELayer


class UniSAGE(torch.nn.Module):
    """Neural network implementation of UniSAGE [1]_ for hypergraph classification.

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
    input_drop: float, default=0.2
        Dropout rate for the input features.
    layer_drop: float, default=0.2
        Dropout rate for the hidden features.
    task_level: str, default="graph"
        Level of the task. Either "graph" or "node".
        If "graph", the output is pooled over all nodes in the hypergraph.

    References
    ----------
    .. [1] Huang and Yang.
        UniGNN: a unified framework for graph and hypergraph neural networks.
        IJCAI 2021.
        https://arxiv.org/pdf/2105.00956.pdf
    """

    def __init__(
        self, 
        in_channels,
        hidden_channels,
        out_channels, 
        input_drop=0.2,
        layer_drop=0.2,
        n_layers=2,
        task_level="graph",
    ):
        super().__init__()

        self.input_drop = torch.nn.Dropout(input_drop)
        self.layer_drop = torch.nn.Dropout(layer_drop)
        

        layers = []
        layers.append(
                UniSAGELayer(
                    in_channels=in_channels,
                    out_channels=hidden_channels,
                )
            )
        for _ in range(n_layers - 1):
            layers.append(
                UniSAGELayer(
                    in_channels=hidden_channels,
                    out_channels=hidden_channels,
                )
            )
        self.layers = torch.nn.ModuleList(layers)
        
        self.linear_out = torch.nn.Linear(hidden_channels, out_channels)
        self.out_pool = True if task_level == "graph" else False

    def forward(self, x_0, incidence_1):
        """Forward computation through layers, then linear layer, then global max pooling.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_edges, channels_edge)
            Edge features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """

        x_0 = self.input_drop(x_0)
        
        # Iterate over layers
        for layer in self.layers:
            x_0 = layer(x_0, incidence_1)
            x_0 = self.layer_drop(x_0)
            x_0 = torch.nn.functional.relu(x_0)
        # Pool over all nodes in the hypergraph 
        if self.out_pool is True:
            x = torch.max(x_0, dim=0)[0]
        else:
            x = x_0

        return self.linear_out(x)

