"""Implementation of a simplified, convolutional version of CCXN layer from Hajij et. al: Cell Complex Neural Networks."""

import torch

from topomodelx.base.conv import Conv


class CCXNLayer(torch.nn.Module):
    """Layer of a Convolutional Cell Complex Network (CCXN).

    Implementation of the CCXN layer proposed in [HIZ20]_.

    This layer is composed of two convolutional layers:
    1. A convolutional layer sending messages from nodes to nodes.
    2. A convolutional layer sending messages from edges to faces.
    Optionally, attention mechanisms can be used.

    Notes
    -----
    This is the architecture proposed for entire complex classification.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes (0-cells).
    in_channels_1 : int
        Dimension of input features on edges (1-cells).
    in_channels_2 : int
        Dimension of input features on faces (2-cells).
    att : bool
        Whether to use attention.

    References
    ----------
    .. [HIZ20] Hajij, Istvan, Zamzmi. Cell Complex Neural Networks.
        Topological Data Analysis and Beyond Workshop at NeurIPS 2020.
        https://arxiv.org/pdf/2010.00743.pdf
    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, att=False):
        super().__init__()
        self.conv_0_to_0 = Conv(
            in_channels=in_channels_0, out_channels=in_channels_0, att=att
        )
        self.conv_1_to_2 = Conv(
            in_channels=in_channels_1, out_channels=in_channels_2, att=att
        )

    def forward(self, x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2, x_2=None):
        """Forward pass.

        The forward pass was initially proposed in [HIZ20]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        References
        ----------
        .. [HIZ20] Hajij, Istvan, Zamzmi. Cell Complex Neural Networks.
            Topological Data Analysis and Beyond Workshop at NeurIPS 2020.
            https://arxiv.org/pdf/2010.00743.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_0_cells, channels]
            Input features on the nodes of the cell complex.
        x_1 : torch.Tensor, shape=[n_1_cells, channels]
            Input features on the edges of the cell complex.
        neighborhood_0_to_0 : torch.sparse
            shape=[n_0_cells, n_0_cells]
            Neighborhood matrix mapping nodes to nodes (A_0_up).
        neighborhood_1_to_2 : torch.sparse
            shape=[n_2_cells, n_1_cells]
            Neighborhood matrix mapping edges to faces (B_2^T).
        x_2 : torch.Tensor, shape=[n_2_cells, channels]
            Input features on the faces of the cell complex.
            Optional, only required if attention is used between edges and faces.

        Returns
        -------
        _ : torch.Tensor, shape=[1, num_classes]
            Output prediction on the entire cell complex.
        """
        x_0 = torch.nn.functional.relu(x_0)
        x_1 = torch.nn.functional.relu(x_1)

        x_0 = self.conv_0_to_0(x_0, neighborhood_0_to_0)
        x_0 = torch.nn.functional.relu(x_0)

        x_2 = self.conv_1_to_2(x_1, neighborhood_1_to_2, x_2)
        x_2 = torch.nn.functional.relu(x_2)

        return x_0, x_1, x_2
