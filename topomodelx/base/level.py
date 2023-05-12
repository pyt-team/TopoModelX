"""Base classes for higher order message passing on topological domains."""

import torch

from topomodelx.base.merge import _Merge
from topomodelx.utils.scatter import scatter


class Level(torch.nn.Module):
    """Level.

    Parameters
    ----------
    message_passings : list or MessagePassing or Merge
        list of MessagePassing and Merge objects that form
        the level within one layer.
    """

    def __init__(self, message_passings):
        super().__init__()
        self.message_passings = message_passings

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : list or Tensor
            Input features.

        Returns
        -------
        _ : list or Tensor
            Output features.
        """
        # the level is a single message passing (a road)
        if not isinstance(x, list):
            if not isinstance(self.message_passings, list):
                return self.message_passings(x)
            # the level is a split: sends one x into several roads
            outputs = []
            for message_passing in self.message_passings:
                outputs.append(message_passing(x))
            return outputs

        if isinstance(self.message_passings, list):
            outputs = []
            for message_passing, x in zip(self.message_passings, x):
                outputs.append(message_passing(x))
            return outputs

        # the level is a merge: sends several x's into a merge
        x = torch.concat(x)
