"""Simplicial Attention Network (SAN) implementation for binary edge classification."""
import torch

from topomodelx.nn.simplicial.san_layer import SANLayer


class SAN(torch.nn.Module):
    r"""Simplicial Attention Network (SAN) implementation for binary edge classification.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    hidden_channels : int
        Dimension of hidden features.
    out_channels : int
        Dimension of output features.
    n_filters : int, default = 2
        Approximation order for simplicial filters.
    order_harmonic : int, default = 5
        Approximation order for harmonic convolution.
    epsilon_harmonic : float, default = 1e-1
        Epsilon value for harmonic convolution.
    n_layers : int, default = 2
        Number of message passing layers.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        n_filters=2,
        order_harmonic=5,
        epsilon_harmonic=1e-1,
        n_layers=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.order_harmonic = order_harmonic
        self.epsilon_harmonic = epsilon_harmonic
        if n_layers == 1:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    n_filters=self.n_filters,
                )
            ]
        else:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.hidden_channels,
                    n_filters=self.n_filters,
                )
            ]
            for _ in range(n_layers - 2):
                self.layers.append(
                    SANLayer(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        n_filters=self.n_filters,
                    )
                )
            self.layers.append(
                SANLayer(
                    in_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    n_filters=self.n_filters,
                )
            )
        self.linear = torch.nn.Linear(out_channels, 2)

    def compute_projection_matrix(self, laplacian):
        """Compute the projection matrix.

        The matrix is used to calculate the harmonic component in SAN layers.

        Parameters
        ----------
        laplacian : torch.Tensor, shape = (n_edges, n_edges)
            Hodge laplacian of rank 1.

        Returns
        -------
        torch.Tensor, shape = (n_edges, n_edges)
            Projection matrix.
        """
        projection_mat = (
            torch.eye(laplacian.shape[0]) - self.epsilon_harmonic * laplacian
        )
        projection_mat = torch.linalg.matrix_power(projection_mat, self.order_harmonic)
        return projection_mat

    def forward(self, x, laplacian_up, laplacian_down):
        """Forward computation.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_nodes, channels_in)
            Node features.
        laplacian_up : torch.Tensor, shape = (n_edges, n_edges)
            Upper laplacian of rank 1.
        laplacian_down : torch.Tensor, shape = (n_edges, n_edges)
            Down laplacian of rank 1.

        Returns
        -------
        torch.Tensor, shape = (n_nodes, 2)
            One-hot labels assigned to edges.
        """
        # Compute the projection matrix for the harmonic component
        laplacian = laplacian_up + laplacian_down
        projection_mat = self.compute_projection_matrix(laplacian)

        # Forward computation
        for layer in self.layers:
            x = layer(x, laplacian_up, laplacian_down, projection_mat)
        return torch.sigmoid(self.linear(x))
