"""Simplicial Complex Convolutional Network Implementation for binary node classification."""
import torch

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


class SCCN(torch.nn.Module):
    """Simplicial Complex Convolutional Network Implementation for binary node classification.

    Parameters
    ----------
    channels : int
        Dimension of features.
    max_rank : int
        Maximum rank of the cells in the simplicial complex.
    n_layers : int
        Number of message passing layers.
    update_func : str
        Activation function used in aggregation layers.
    """

    def __init__(self, channels, max_rank, n_layers=2, update_func="sigmoid"):
        super().__init__()
        layers = []
        for _ in range(n_layers):
            layers.append(
                SCCNLayer(
                    channels=channels,
                    max_rank=max_rank,
                    update_func=update_func,
                )
            )
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, features, incidences, adjacencies):
        """Forward computation.

        Parameters
        ----------
        features : dict[int, torch.Tensor], length=max_rank+1, shape = (n_rank_r_cells, channels)
            Input features on the cells of the simplicial complex.
        incidences : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_minus_1_cells, n_rank_r_cells)
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adjacencies : dict[int, torch.sparse], length=max_rank, shape = (n_rank_r_cells, n_rank_r_cells)
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.

        Returns
        -------
        Dict of torch.Tensor
            rank_0 : torch.Tensor
                Output features on nodes.
            rank_1 : torch.Tensor
                Output features on edges.
            rank_2 : torch.Tensor
                Output features on triangles.
            rank_3 : torch.Tensor
                Output features on tetrahedra.
            ...
            (up to max_rank)
        """
        for layer in self.layers:
            features = layer(features, incidences, adjacencies)
        return features
