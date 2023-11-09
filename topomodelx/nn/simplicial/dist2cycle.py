"""Dist2Cycle model for binary node classification."""
import torch

from topomodelx.nn.simplicial.dist2cycle_layer import Dist2CycleLayer


class Dist2Cycle(torch.nn.Module):
    """High Skip Network Implementation for binary node classification.

    Parameters
    ----------
    channels : int
        Dimension of features.
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
        self.layers = layers

    def forward(self, x_1e, Linv, adjacency):
        """Forward computation.

        Parameters
        ----------
        x_1e : torch.Tensor, shape = (n_nodes, channels)
            Node features.
        Linv : torch.Tensor
        adjacency : torch.Tensor

        Returns
        -------
        torch.Tensor, shape = (n_nodes, channels)
            Final node hidden representations.
        """
        for layer in self.layers:
            x_1e = layer(x_1e, Linv, adjacency)
        return x_1e
