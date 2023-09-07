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
    simplex_order_k : int
        Order r of the considered simplices. Default to 1 (edges).
    num_filters_J : int, optional
        Approximation order for simplicial filters. Defaults to 2.
    J_har : int, optional
        Approximation order for harmonic convolution. Defaults to 5.
    epsilon_har : float, optional
        Epsilon value for harmonic convolution. Defaults to 1e-1.
    n_layers : int, optional
        Number of message passing layers. Defaults to 2.
    """

    def __init__(
        self,
        in_channels,
        hidden_channels,
        out_channels,
        num_filters_J=2,
        J_har=5,
        epsilon_har=1e-1,
        n_layers=2,
    ):
        super().__init__()
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels
        self.num_filters_J = num_filters_J
        self.J_har = J_har
        self.epsilon_har = epsilon_har
        if n_layers == 1:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.out_channels,
                    num_filters_J=self.num_filters_J,
                )
            ]
        else:
            self.layers = [
                SANLayer(
                    in_channels=self.in_channels,
                    out_channels=self.hidden_channels,
                    num_filters_J=self.num_filters_J,
                )
            ]
            for _ in range(n_layers - 2):
                self.layers.append(
                    SANLayer(
                        in_channels=self.hidden_channels,
                        out_channels=self.hidden_channels,
                        num_filters_J=self.num_filters_J,
                    )
                )
            self.layers.append(
                SANLayer(
                    in_channels=self.hidden_channels,
                    out_channels=self.out_channels,
                    num_filters_J=self.num_filters_J,
                )
            )
        self.linear = torch.nn.Linear(out_channels, 2)

    def compute_projection_matrix(self, L):
        """Compute the projection matrix used to calculate the harmonic component in SAN layers.

        Parameters
        ----------
        L : tensor
            shape = [n_edges, n_edges]
            Hodge laplacian of rank 1.

        Returns
        -------
        _ : tensor
            shape = [n_edges, n_edges]
            Projection matrix.
        """
        P = torch.eye(L.shape[0]) - self.epsilon_har * L
        P = torch.linalg.matrix_power(P, self.J_har)
        return P

    def forward(self, x, Lup, Ldown):
        """Forward computation.

        Parameters
        ----------
        x : tensor
            shape = [n_nodes, channels_in]
            Node features.

        Lup : tensor
            shape = [n_edges, n_edges]
            Upper laplacian of rank 1.

        Ld : tensor
            shape = [n_edges, n_edges]
            Down laplacian of rank 1.


        Returns
        -------
        _ : tensor
            shape = [n_nodes, 2]
            One-hot labels assigned to edges.

        """
        # Compute the projection matrix for the harmonic component
        L = Lup + Ldown
        P = self.compute_projection_matrix(L)

        # Forward computation
        for layer in self.layers:
            x = layer(x, Lup, Ldown, P)
        return torch.sigmoid(self.linear(x))
