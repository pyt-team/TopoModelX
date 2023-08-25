"""Simplex Convolutional Network (SCN) Layer [Yang et al. LoG 2022]."""
import torch

from topomodelx.base.conv import Conv


class SCN2Layer(torch.nn.Module):
    """Layer of a Simplex Convolutional Network (SCN).

    Implementation of the SCN layer proposed in [YSB22]_ for a simplicial complex of
    rank 2, that is for 0-cells (nodes), 1-cells (edges) and 2-cells (faces) only.

    This layer corresponds to the rightmost tensor diagram labeled Yang22c in
    Figure 11 of [PSHM23]_.

    See Also
    --------
    topomodelx.nn.simplicial.sccn_layer.SCCNLayer : SCCN layer
        Simplicial Complex Convolutional Network (SCCN) layer proposed in [YSB22]_.
        The difference between SCCN and SCN is that:
        - SCN passes messages between cells of the same rank,
        - SCCN passes messages between cells of the same ranks, one rank above
        and one rank below.

    Notes
    -----
    This architecture is proposed for simplicial complex classification.

    References
    ----------
    .. [YSB22] Ruochen Yang, Frederic Sala, and Paul Bogdan.
        Efficient Representation Learning for Higher-Order Data with
        Simplicial Complexes. In Bastian Rieck and Razvan Pascanu, editors,
        Proceedings of the First Learning on Graphs Conference, volume 198
        of Proceedings of Machine Learning Research, pages 13:1–13:21. PMLR,
        09–12 Dec 2022a. https://proceedings.mlr.press/v198/yang22a.html.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2):
        super().__init__()
        self.conv_0_to_0 = Conv(in_channels=in_channels_0, out_channels=in_channels_0)
        self.conv_1_to_1 = Conv(in_channels=in_channels_1, out_channels=in_channels_1)
        self.conv_2_to_2 = Conv(in_channels=in_channels_2, out_channels=in_channels_2)

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_0_to_0.reset_parameters()
        self.conv_1_to_1.reset_parameters()
        self.conv_2_to_2.reset_parameters()

    def forward(self, x_0, x_1, x_2, laplacian_0, laplacian_1, laplacian_2):
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, node_features]
            Input features on the nodes of the simplicial complex.
        x_1 : torch.Tensor, shape=[n_edges, edge_features]
            Input features on the edges of the simplicial complex.
        x_2 : torch.Tensor, shape=[n_faces, face_features]
            Input features on the faces of the simplicial complex.
        laplacian_0 : torch.sparse, shape=[n_nodes, n_nodes]
            Normalized Hodge Laplacian matrix = L_upper + L_lower.
        laplacian_1 : torch.sparse, shape=[n_edges, n_edges]
            Normalized Hodge Laplacian matrix.
        laplacian_2 : torch.sparse, shape=[n_faces, n_faces]
            Normalized Hodge Laplacian matrix.

        Returns
        -------
        _ : torch.Tensor, shape=[n_nodes, channels]
            Output features on the nodes of the simplicial complex.
        """
        x_0 = self.conv_0_to_0(x_0, laplacian_0)
        x_0 = torch.nn.functional.relu(x_0)
        x_1 = self.conv_1_to_1(x_1, laplacian_1)
        x_1 = torch.nn.functional.relu(x_1)
        x_2 = self.conv_2_to_2(x_2, laplacian_2)
        x_2 = torch.nn.functional.relu(x_2)
        return x_0, x_1, x_2
