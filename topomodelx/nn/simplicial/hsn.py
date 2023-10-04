"""High Skip Network Implementation model for binary node classification."""
import torch

from topomodelx.nn.simplicial.hsn_layer import HSNLayer


class HSN(torch.nn.Module):
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
                HSNLayer(
                    channels=channels,
                )
            )
        self.linear = torch.nn.Linear(channels, 2)
        self.layers = layers

    def forward(self, x_0, incidence_1, adjacency_0):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, channels)
            Node features.

        incidence_1 : torch.Tensor, shape = (n_nodes, n_edges)
            Boundary matrix of rank 1.

        adjacency_0 : torch.Tensor, shape = (n_nodes, n_nodes)
            Adjacency matrix (up) of rank 0.

        Returns
        -------
        torch.Tensor, shape = (n_nodes, 2)
            One-hot labels assigned to nodes.

        """
        for layer in self.layers:
            x_0 = layer(x_0, incidence_1, adjacency_0)
        logits = self.linear(x_0)
        return torch.softmax(logits, dim=-1)
