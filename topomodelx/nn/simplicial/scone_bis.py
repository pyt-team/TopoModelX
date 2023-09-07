"""Neural network implementation of classification using SCoNe."""
import torch

from topomodelx.nn.simplicial.scone_layer_bis import SCoNeLayer


class SCoNeNN(torch.nn.Module):
    """Neural network implementation of classification using SCoNe.

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
                SCoNeLayer(
                    channels=channels,
                )
            )
        self.linear = torch.nn.Linear(channels, 2)
        self.layers = layers

    def forward(self, x_1, up_lap1, down_lap1, iden):
        """Forward computation.

        Parameters
        ----------
        x_0 : tensor
            shape = [n_nodes, channels]
            Node features.

        up_lap1 : tensor
            shape = [n_edges, n_edges]
            Upper Laplacian matrix of rank 1.

        down_lap1 : tensor
            shape = [n_edges, n_edges]
            Laplacian matrix (down) of rank 1.

        Returns
        -------
        _ : tensor
            shape = [n_nodes, 2]
            One-hot labels assigned to nodes.
        """
        for layer in self.layers:
            x_1 = layer(x_1, up_lap1, down_lap1, iden)
        x_1 = self.linear(x_1)
        return torch.softmax(x_1, dim=-1)
