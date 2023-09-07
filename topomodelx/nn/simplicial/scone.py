"""Neural network implementation of classification using SCoNe."""
import numpy as np
import torch
from scipy.spatial import Delaunay, distance
from torch import nn
from torch.utils.data.dataset import Dataset

from topomodelx.nn.simplicial.scone_layer import SCoNeLayer
from toponetx.classes.simplicial_complex import SimplicialComplex


def generate_complex(N: int = 100) -> tuple[SimplicialComplex, np.ndarray]:
    """
    Generate a simplicial complex of dimension 2 as follows:.

        1. Uniformly sample N random points form the unit square and build the Delaunay triangulation.
        2. Delete triangles contained in some pre-defined disks.
    """
    points = np.random.uniform(0, 1, size=(N, 2))

    # Sort points by the sum of their coordinates
    c = np.sum(points, axis=1)
    order = np.argsort(c)
    points = points[order]

    tri = Delaunay(points)  # Create Delaunay triangulation

    # Remove triangles having centroid inside the disks
    disk_centers = np.array([[0.3, 0.7], [0.7, 0.3]])
    disk_radius = 0.15
    simplices = []
    indices_included = set()
    for simplex in tri.simplices:
        center = np.mean(points[simplex], axis=0)
        if ~np.any(distance.cdist([center], disk_centers) <= disk_radius, axis=1):
            # Centroid is not contained in some disk, so include it.
            simplices.append(simplex)
            indices_included |= set(simplex)

    # Re-index vertices before constructing the simplicial complex
    idx_dict = {i: j for j, i in enumerate(indices_included)}
    for i in range(len(simplices)):
        for j in range(3):
            simplices[i][j] = idx_dict[simplices[i][j]]

    sc = SimplicialComplex(simplices)
    coords = points[list(indices_included)]
    return sc, coords


class TrajectoriesDataset(Dataset):
    """Create a dataset of trajectories."""

    def __init__(self, sc: SimplicialComplex, trajectories: list[list[int]]) -> None:
        self.trajectories = trajectories
        self.sc = sc
        self.adjacency = torch.Tensor(sc.adjacency_matrix(0).toarray())

        # Lookup table used to speed up vectorizing of trajectories
        self.edge_lookup_table = {}
        for i, edge in enumerate(self.sc.skeleton(1)):
            self.edge_lookup_table[edge] = (1, i)
            self.edge_lookup_table[edge[::-1]] = (-1, i)

    def __getitem__(
        self, index: int
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Get the index of the trajectory and its neighbors."""
        trajectory = self.trajectories[index]
        vectorized_trajectory = self.vectorize_path(
            trajectory[:-1]
        )  # Discard the last node

        # Find neighbors of the last node in the trajectory (for use in the forward pass of SCoNe)
        neighbors_mask = (
            torch.Tensor(self.adjacency[trajectory[-2]] > 0).float().unsqueeze(-1)
        )

        last_node = torch.tensor(trajectory[-1])

        return vectorized_trajectory, neighbors_mask, last_node

    def __len__(self) -> int:
        """Trajectories in the dataset."""
        return len(self.trajectories)

    def vectorize_path(self, path: list[int]) -> torch.Tensor:
        """Vectorize a path of nodes into a vector representation of the trajectory."""
        # Create a vector representation of a trajectory.
        m = len(self.sc.skeleton(1))
        c0 = torch.zeros((m, 1))
        for j in range(0, len(path) - 1):
            edge = (path[j], path[j + 1])
            sign, i = self.edge_lookup_table[edge]
            c0[i] = sign
        return c0


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
