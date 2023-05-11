"""Base classes for higher order message passing on topological domains."""

import torch

from topomodelx.base.merge import _Merge
from topomodelx.utils.scatter import scatter


class _Level(torch.nn.Module):
    def ___init___(self, message_passings, inter_aggr):
        """
        Parameters
        ----------
        message_passings : list of MessagePassing and Merge objects
            TODO.
        inter_aggr : string
            Aggregation method.
            (Inter-neighborhood).
        update_func : string
            Update function.
        """
        super(_Level, self).__init__()
        self.message_passings = message_passings
        self.inter_aggr = inter_aggr

    def forward(self, x):
        outputs = []
        for mp in self.message_passings:
            if not isinstance(mp, list):
                outputs.append(mp.forward(x))
            else:
                merge = _Merge(mp, inter_aggr="sum")
                outputs.append(merge.forward(x))
        return outputs
