"""Dist2Cycle model for binary node classification."""
import torch

from topomodelx.nn.simplicial.dist2cycle_layer import Dist2CycleLayer


class Dist2Cycle(torch.nn.Module):
    """High Skip Network Implementation for binary node classification.

    Parameters
    ----------
    channels : int
        Dimension of features
    n_layers : int
        Amount of message passing layers.

    """

    def __init__(self, channels, n_layers=2):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                Dist2CycleLayer(
                    channels=channels,
                )
            )
        self.linear = torch.nn.Linear(channels, 2)  # changed
        self.layers = layers

    def forward(self, x_1e, Linv, adjacency):
        """Forward computation.

        Parameters
        ----------
        x_0 : tensor
            shape = [n_nodes, channels]
            Node features.

        incidence_1 : tensor
            shape = [n_nodes, n_edges]
            Boundary matrix of rank 1.

        adjacency_0 : tensor
            shape = [n_nodes, n_nodes]
            Adjacency matrix (up) of rank 0.

        Returns
        -------
        _ : tensor
            shape = [n_nodes, 2]
            One-hot labels assigned to nodes.

        """
        for layer in self.layers:
            x_1 = layer(x_1e, Linv, adjacency)
        logits = self.linear(x_1)
        return torch.softmax(logits, dim=-1)
