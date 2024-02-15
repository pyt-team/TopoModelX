"""CCXN class."""

import torch

from topomodelx.nn.cell.ccxn_layer import CCXNLayer


class CCXN(torch.nn.Module):
    """CCXN [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes.
    in_channels_1 : int
        Dimension of input features on edges.
    in_channels_2 : int
        Dimension of input features on faces.
    n_layers : int
        Number of CCXN layers.
    att : bool
        Whether to use attention.

    References
    ----------
    .. [1] Hajij, Istvan, Zamzmi.
        Cell complex neural networks.
        Topological data analysis and beyond workshop at NeurIPS 2020.
        https://arxiv.org/pdf/2010.00743.pdf
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        n_layers=2,
        att=False,
    ):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                CCXNLayer(
                    in_channels_0=in_channels_0,
                    in_channels_1=in_channels_1,
                    in_channels_2=in_channels_2,
                    att=att,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2):
        """Forward computation through layers.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        neighborhood_0_to_0 : torch.Tensor, shape = (n_nodes, n_nodes)
            Adjacency matrix of rank 0 (up).
        neighborhood_1_to_2 : torch.Tensor, shape = (n_faces, n_edges)
            Transpose of boundary matrix of rank 2.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Final hidden states of the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Final hidden states the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_edges, in_channels_2)
            Final hidden states of the faces (2-cells).
        """
        for layer in self.layers:
            x_0, x_1, x_2 = layer(x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2)
        return x_0, x_1, x_2
