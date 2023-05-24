"""Implementation of a simplified, convolutional version of CXN layer from Hajij et. al: Cell Complex Neural Networks."""

import torch

from topomodelx.base.conv import Conv


class CXNLayer(torch.nn.Module):
    """Layer of a simplified CXN.

    Implementation of a convolutional version of the CXN layer (no attention)
    from the paper by Hajij et. al : Cell Complex Neural Networks
    https://arxiv.org/pdf/2010.00743.pdf
    Note: this is the architecture proposed for entire complex classification.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on nodes.
    in_channels_1 : int
        Dimension of input features on edges.
    in_channels_2 : int
        Dimension of input features on faces.
    """

    def __init__(self, in_channels_0, in_channels_1, in_channels_2, att=False):
        super().__init__()
        self.conv_0_to_0 = Conv(in_channels_0, in_channels_0, att=att)
        self.conv_1_to_2 = Conv(in_channels_1, in_channels_2)

    def forward(self, x_0, x_1, neighborhood_0_to_0, neighborhood_1_to_2):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.tensor
            shape=[n_0_cells, channels]
            Input features on the nodes of the cell complex.
        x_1 : torch.tensor
            shape=[n_1_cells, channels]
            Input features on the edges of the cell complex.
        neighborhood_0_to_0 : torch.sparse
            shape=[n_0_cells, n_0_cells]
            Neighborhood matrix mapping nodes to nodes (A_0_up).
        neighborhood_1_to_2 : torch.sparse
            shape=[n_2_cells, n_1_cells]
            Neighborhood matrix mapping edges to faces (B_2^T).

        Returns
        -------
        _ : torch.tensor
            shape=[1, num_classes]
            Output prediction on the entire cell complex.
        """
        x_0 = torch.nn.functional.relu(x_0)
        x_1 = torch.nn.functional.relu(x_1)

        x_0 = self.conv_0_to_0(x_0, neighborhood_0_to_0)
        x_0 = torch.nn.functional.relu(x_0)

        x_2 = self.conv_1_to_2(x_1, neighborhood_1_to_2)
        x_2 = torch.nn.functional.relu(x_2)

        return x_0, x_1, x_2
