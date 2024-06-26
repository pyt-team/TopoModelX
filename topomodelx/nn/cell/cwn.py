"""CWN class."""

import torch
import torch.nn.functional as F

from topomodelx.nn.cell.cwn_layer import CWNLayer


class CWN(torch.nn.Module):
    """Implementation of a specific version of CW network [1]_.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    hid_channels : int
        Dimension of hidden features.
    n_layers : int
        Number of CWN layers.
    **kwargs : optional
        Additional arguments CWNLayer.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        hid_channels,
        n_layers,
        **kwargs,
    ):
        super().__init__()

        self.proj_0 = torch.nn.Linear(in_channels_0, hid_channels)
        self.proj_1 = torch.nn.Linear(in_channels_1, hid_channels)
        self.proj_2 = torch.nn.Linear(in_channels_2, hid_channels)

        self.layers = torch.nn.ModuleList(
            CWNLayer(
                in_channels_0=hid_channels,
                in_channels_1=hid_channels,
                in_channels_2=hid_channels,
                out_channels=hid_channels,
                **kwargs,
            )
            for _ in range(n_layers)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        incidence_2,
        incidence_1_t,
    ):
        """Forward computation through projection, convolutions, linear layers and average pooling.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Input features on the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Input features on the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_faces, in_channels_2)
            Input features on the faces (2-cells).
        adjacency_0 : torch.Tensor, shape = (n_edges, n_edges)
            Upper-adjacency matrix of rank 1.
        incidence_2 : torch.Tensor, shape = (n_edges, n_faces)
            Boundary matrix of rank 2.
        incidence_1_t : torch.Tensor, shape = (n_edges, n_nodes)
            Coboundary matrix of rank 1.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, in_channels_0)
            Final hidden states of the nodes (0-cells).
        x_1 : torch.Tensor, shape = (n_edges, in_channels_1)
            Final hidden states the edges (1-cells).
        x_2 : torch.Tensor, shape = (n_edges, in_channels_2)
            Final hidden states of the faces (2-cells).
        """
        x_0 = F.elu(self.proj_0(x_0))
        x_1 = F.elu(self.proj_1(x_1))
        x_2 = F.elu(self.proj_2(x_2))

        for layer in self.layers:
            x_1 = layer(
                x_0,
                x_1,
                x_2,
                adjacency_0,
                incidence_2,
                incidence_1_t,
            )

        return x_0, x_1, x_2
