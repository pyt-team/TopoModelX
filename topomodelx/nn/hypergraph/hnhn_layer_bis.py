"""HNHN (Hypergraph Network with Hyperedge Neurons) Layer introduced in Dong et al. 2020."""

import torch
from torch.nn import functional as F


class HNHNLayer(torch.nn.Module):
    """HNHN Layer [1]_.

    Given the input representation of nodes, this layer returns a new representation using hyperedges
    as a relay. In other words it makes a intermediary representation for hyperedges with those of the nodes
    and then makes the final representation of the nodes using the representation of the hyperedges.
    During building the representation for a hyperedge/node, this layer multiplies a normalized weight to the
    features of the neighboring nodes/hyperedges, reflecting different levels of importance across the neighbors.

    Parameters
    ----------
    in_features : int
        The input dimension of node features.
    incidence_1 : torch.sparse.Tensor, shape = (n_nodes, n_edges)
        Incidence matrix mapping hyperedges to nodes (B_1).
    activation_func: Callable
        Called on the new representations.
    normalization_param_alpha: float
        The param that weights multiplied into hyperedge representations are powered to beforehand.
    normalization_param_beta: float
        The param that weights multiplied into node representations are powered to beforehand.

    References
    ----------
    .. [1] Dong, Sawin, Bengio.
        HNHN: hypergraph networks with hyperedge neurons.
        Graph Representation Learning and Beyond Workshop at ICML 2020.
        https://grlplus.github.io/papers/40.pdf
    """

    def __init__(
        self,
        in_features,
        incidence_1,
        activation_func=F.relu,
        normalization_param_alpha: float = 0.0,
        normalization_param_beta: float = 0.0,
    ) -> None:
        super().__init__()

        incidence_1 = incidence_1.to(torch.float)

        d_E = incidence_1.sum(dim=0).to_dense()
        d_E_powered = d_E**normalization_param_alpha
        # In HNHN layer, it is assumed that a node is attached to at least one hyperedge, so the
        # division below is valid.
        normalizing_factor = 1 / torch.sparse.mm(incidence_1, d_E_powered.view(-1, 1))
        self.weighted_hyperedge_to_node_incidence = (
            normalizing_factor * incidence_1 * d_E_powered
        )

        d_V = incidence_1.sum(dim=1).to_dense()
        d_V_powered = d_V**normalization_param_beta
        incidence_1_T = incidence_1.transpose(1, 0)
        normalizing_factor = 1 / torch.sparse.mm(incidence_1_T, d_V_powered.view(-1, 1))
        self.weighted_node_to_hyperedge_incidence = (
            normalizing_factor * incidence_1_T * d_V_powered
        )

        self.node_linear_transform = torch.nn.Linear(in_features, in_features)
        self.hyperedge_linear_transform = torch.nn.Linear(in_features, in_features)

        self.activation = activation_func

    def forward(self, x_0):
        r"""Forward computation.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_nodes, in_features)
            Input features of the nodes.

        Returns
        -------
        x_0 : torch.Tensor, shape = (n_nodes, in_features)
            Output features of the nodes.
        x_1 : torch.Tensor, shape = (n_hyperedges, in_features)
            Output features of the hyperedges.
        """
        x_1_prime = self.activation(
            self.hyperedge_linear_transform(
                self.weighted_node_to_hyperedge_incidence @ x_0
            )
        )
        x_0_prime = self.activation(
            self.node_linear_transform(
                self.weighted_hyperedge_to_node_incidence @ x_1_prime
            )
        )

        return x_0_prime, x_1_prime
