"""Neural network implementation of classification using SCoNe."""
import torch
from torch import nn

from topomodelx.nn.simplicial.scone_layer import SCoNeLayer
from toponetx.nn.simplicial.simplicial_complex import SimplicialComplex


class SCoNe(nn.Module):
    """Neural network implementation of classification using SCoNe."""

    def __init__(self, sc: SimplicialComplex, hidden_dims: list[int]) -> None:
        super().__init__()
        self.incidence_1 = torch.Tensor(sc.incidence_matrix(1).toarray())
        self.incidence_2 = torch.Tensor(sc.incidence_matrix(2).toarray())
        self.adjacency = torch.Tensor(sc.adjacency_matrix(0).toarray())

        # Weights for the last layer
        self.weights = nn.parameter.Parameter(torch.Tensor(hidden_dims[-1], 1))
        nn.init.xavier_uniform_(self.weights)

        self.hidden_dimensions = hidden_dims
        self.L = len(hidden_dims)

        # Stack multiple SCoNe layers with given hidden dimensions
        self.layers = nn.ModuleList([SCoNeLayer(1, hidden_dims[0])])
        for i in range(self.L - 1):
            self.layers.append(SCoNeLayer(hidden_dims[i], hidden_dims[i + 1]))

        # Initialize parameters
        for layer in self.layers:
            layer.reset_parameters()

    def forward(self, x: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """Forward pass through the network."""
        for layer in self.layers:
            x = layer(x, self.incidence_1, self.incidence_2)
        # Last layer going from edges to nodes using the boundary operator
        x = self.incidence_1 @ x @ self.weights
        # Take softmax only over neighbors by setting the logits of non-neighbors to approximately -inf
        x = x + (1e-15 + mask).log()
        x = nn.functional.log_softmax(x, dim=1)
        return x
