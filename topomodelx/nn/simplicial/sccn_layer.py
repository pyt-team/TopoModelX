"""Simplicial Complex Convolutional Network Layer. [Yang et al. LoG 2022]"""
from typing import Dict

import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCCNLayer(torch.nn.Module):
    """
    TODO: more documentation.

    Parameters
    ----------
    channels : int
        Dimension of features on each simplicial cell.
    max_rank : int
        Maximum rank of the cells in the simplicial complex.
    """

    def __init__(
        self,
        channels,
        max_rank,
    ):
        super().__init__()
        self.channels = channels
        self.max_rank = max_rank

        # convolutions within the same rank
        self.convs_same_rank = torch.nn.ModuleDict(
            {
                f'rank_{rank}': Conv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=None,
                )
                for rank in range(max_rank + 1)
            }
        )

        # convolutions from lower to higher rank
        self.convs_low_to_high = torch.nn.ModuleDict(
            {
                f'rank_{rank}': Conv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=None,
                )
                for rank in range(1, max_rank + 1)
            }
        )

        # convolutions from higher to lower rank
        self.convs_high_to_low = torch.nn.ModuleDict(
            {
                f'rank_{rank}': Conv(
                    in_channels=channels,
                    out_channels=channels,
                    update_func=None,
                )
                for rank in range(max_rank)
            }
        )

        # aggregation functions
        self.aggregations = torch.nn.ModuleDict(
            {
                f'rank_{rank}': Aggregation(aggr_func="sum", update_func="sigmoid")
                for rank in range(max_rank + 1)
            }
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        for rank in self.convs_same_rank:
            self.convs_same_rank[rank].reset_parameters()
        for rank in self.convs_low_to_high:
            self.convs_low_to_high[rank].reset_parameters()
        for rank in self.convs_high_to_low:
            self.convs_high_to_low[rank].reset_parameters()

    def forward(self, features, incidences, adjacencies):
        r"""Forward pass.

        The forward pass was initially proposed in [YSB22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        .. math::
            \begin{align*}
            TODO
            \end{align*}

        References
        ----------
        .. [YSB22] Yang, Sala, Bogdan.
            Efficient Representation Learning for Higher-Order Data with Simplicial Complexes.
            https://proceedings.mlr.press/v198/yang22a.html
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        features: Dict[int, torch.Tensor],
                length=max_rank+1,
                shape=[n_rank_r_cells, channels]
            Input features on the cells of the simplicial complex.
        incidences : Dict[int, torch.sparse],
                length=max_rank,
                shape=[n_rank_r_cells, n_rank_r_minus_1_cells]
            Incidence matrices :math:`B_r` mapping r-cells to (r-1)-cells.
        adjacencies : Dict[int, torch.sparse],
                length=max_rank,
                shape=[n_rank_r_cells, n_rank_r_cells]
            Adjacency matrices :math:`H_r` mapping cells to cells
                via lower and upper cells.

        Returns
        -------
        _ : List[torch.Tensor],
                length=max_rank+1,
                shape=[n_rank_r_cells, channels]
            Output features on the cells of the simplicial complex.
        """
        out_features = {}
        for rank in range(self.max_rank + 1):
            list_to_be_aggregated = [
                self.convs_same_rank[f'rank_{rank}'](
                    features[f'rank_{rank}'],
                    adjacencies[f'rank_{rank}'],
                )
            ]
            if rank < self.max_rank:
                list_to_be_aggregated.append(
                    self.convs_high_to_low[f'rank_{rank}'](
                        features[f'rank_{rank+1}'],
                        incidences[f'rank_{rank+1}'],
                    )
                )
            if rank > 0:
                list_to_be_aggregated.append(
                    self.convs_low_to_high[f'rank_{rank}'](
                        features[f'rank_{rank-1}'],
                        incidences[f'rank_{rank}'].transpose(1, 0),
                    )
                )

            out_features[f'rank_{rank}'] = (
                self.aggregations[f'rank_{rank}'](list_to_be_aggregated)
            )

        return out_features
