"""SCA with CMPS."""
import torch

from topomodelx.nn.simplicial.sca_cmps_layer import SCACMPSLayer


class SCACMPS(torch.nn.Module):
    """SCA with CMPS.

    Parameters
    ----------
    in_channels_all : list[int]
        Dimension of features on each node, edge, simplex, tetahedron,... respectively
    complex_dim : int
        Highest dimension of simplicial complex feature being trained on.
    n_layers : int, default = 2
        Amount of message passing layers.
    att : bool
        Whether to use attention.
    """

    def __init__(
        self,
        in_channels_all,
        complex_dim,
        n_layers=2,
        att=False,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.in_channels_all = in_channels_all

        layers = []
        for _ in range(n_layers):
            layers.append(SCACMPSLayer(in_channels_all, complex_dim, att))

        self.layers = torch.nn.ModuleList(layers)

    def forward(self, x, laplacian_down_list, incidence_t_list):
        """Forward computation through layers, then linear layers, then avg pooling.

        Parameters
        ----------
        x : list[torch.Tensor]
            Tensor inputs for each dimension of the complex (nodes, edges, etc.).
        laplacian_down_list : list[torch.Tensor]
            List of the down laplacian matrix for each dimension in the complex starting at edges.
        incidence_t_list : list[torch.Tensor]
            List of the transpose incidence matrices for the edges and faces.

        Returns
        -------
        torch.Tensor, shape = (1)
            Label assigned to whole complex.
        """
        for i in range(self.n_layers):
            x = self.layers[i](x, laplacian_down_list, incidence_t_list)

        return x
