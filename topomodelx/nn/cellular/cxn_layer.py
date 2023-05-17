"""Implementation of a CXN layer from the paper by Hajij et. al: Cell Complex Neural Networks."""
import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.base.conv import Conv


class CXNLayer(nn.Module):
    """Layer of a CXN.

    Implementation of the CXN layer from the paper by Hajij et. al : Cell Complex Neural Networks
    https://arxiv.org/pdf/2010.00743.pdf
    Note: this is the architecture proposed for entire complex classification.


    """

    def __init__(
        self,
        channels,
        num_classes,
    ):
        super().__init__()
        self.conv_1 = Conv(channels, channels)
        self.conv_2 = Conv(channels, channels)
        self.conv_3 = Conv(channels, channels)
        self.lin1 = nn.Linear(channels, num_classes)
        self.lin2 = nn.Linear(channels, num_classes)
        self.lin3 = nn.Linear(channels, num_classes)

    def forward(self, x_0, x_a_1, x_b_1, neighborhood_0_to_0, neighborhood_1_to_2):
        """Forward computation.

        Parameters
        ----------
        x_0 : torch.tensor
            shape=[n_0_cells, channels]
            Input features on the nodes of the cellular complex.
        x_a_1 : torch.tensor
            shape=[n_1_cells, channels]
            First set of input features on the edges of the cellular complex.
        x_b_1 : torch.tensor
            shape=[n_1_cells, channels]
            Second set of input features on the edges of the cellular complex.
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
            Output prediction on the entire cellular complex.
        """
        x_0 = torch.nn.functional.relu(x_0)
        x_a_1 = torch.nn.functional.relu(x_a_1)
        x_b_1 = torch.nn.functional.relu(x_b_1)

        x_0 = self.conv_1(x_0, neighborhood_0_to_0)
        x_0 = torch.nn.functional.relu(x_0)

        x_a_2 = self.conv_2(x_a_1, neighborhood_1_to_2)
        x_b_2 = self.conv_3(x_b_1, neighborhood_1_to_2)
        x_a_2 = torch.nn.functional.relu(x_a_2)
        x_b_2 = torch.nn.functional.relu(x_b_2)

        x_a_2 = self.lin1(x_a_2)
        x_b_2 = self.lin2(x_b_2)
        x_0 = self.lin3(x_0)

        return (
            torch.mean(x_a_2, dim=0) + torch.mean(x_b_2, dim=0) + torch.mean(x_0, dim=0)
        )
