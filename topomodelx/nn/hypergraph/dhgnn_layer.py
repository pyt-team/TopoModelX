"""Implementation of a simplified, convolutional version of DHGNN layer from Jiang et. al: Dynamic Hypergraph Neural Networks."""
import torch

from topomodelx.base.conv import Conv


class DHGNNLayer(torch.nn.Module):
    """Layer of a Dynamic Hypergraph Neural Network (DHGNN)

    A simplified 2-step message passing layer proposed in [JIA19]

    This layer is composed of two convolutional layers:
    1. A convolutional layer sending messages from edges to nodes.
    2. A convolutional layer sending messages from nodes to a new centroid node.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    intermediate_channels : int
        Dimension of intermediate features.
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
    ):
        super().__init__()

        self.layer1_2_to_1 = Conv(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            aggr_norm=False,
            update_func="relu",
        )
        self.layer2_1_to_0 = Conv(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )
        self.linear = torch.nn.Linear(intermediate_channels, 1)

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.layer1_2_to_1.reset_parameters()
        self.layer2_1_to_0.reset_parameters()

    def forward(self, x, incidence_1):
        r"""Forward computation.

        Parameters
        ----------
        x : torch.Tensor, shape=[n_edges, in_channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x : torch.Tensor, shape=[n_edges, 1]
            Output features on the edges of the simplicial complex.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)
        x = self.layer1_2_to_1(x, incidence_1)
        x = torch.mean(self.layer2_1_to_0(x, incidence_1_transpose), dim=0)[0]
        return x
