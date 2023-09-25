"""Simplicial Complex Convolutional Network Implementation for binary node classification."""
import torch

from topomodelx.nn.simplicial.sccn_layer import SCCNLayer


class SCCN(torch.nn.Module):
    """Simplicial Complex Convolutional Network Implementation for binary node classification.

    Parameters
    ----------
    channels : int
        Dimension of features
    max_rank : int
        Maximum rank of the cells in the simplicial complex.
    n_layers : int
        Number of message passing layers.
    n_classes : int
        Number of classes.
    update_func : str
        Activation function used in aggregation layers.

    """

    def __init__(
        self, channels, max_rank, n_layers=2, n_classes=2, update_func="sigmoid"
    ):
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

        assert n_classes >= 2, "n_classes must be >= 2"

        if n_classes == 2:
            n_classes = 1

        self.linear = torch.nn.Linear(channels, n_classes)
        self.layers = torch.nn.ModuleList(layers)

    def forward(self, features, incidences, adjacencies):
        """Forward computation.

        Parameters
        ----------
        features: Dict[int, torch.Tensor], length=max_rank+1, shape=[n_rank_r_cells, channels]
            Input features on the cells of the simplicial complex.
        incidences : Dict[int, torch.sparse], length=max_rank, shape=[n_rank_r_minus_1_cells, n_rank_r_cells]
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adjacencies : Dict[int, torch.sparse], length=max_rank, shape=[n_rank_r_cells, n_rank_r_cells]
            Adjacency matrices :math:`H_r` mapping cells to cells via lower and upper cells.

        Returns
        -------
        _ : tensor
            If n_classes > 2:
                shape = [n_nodes, n_classes]
                Logits assigned to each node.
            If n_classes == 2:
                shape = [n_nodes,]
                Binary logits assigned to each node.

        """
        for layer in self.layers:
            features = layer(features, incidences, adjacencies)
        logits = self.linear(features["rank_0"]).squeeze()
        return logits
