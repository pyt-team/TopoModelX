"""Aggregation module."""
from typing import Literal

import torch


class Aggregation(torch.nn.Module):
    """Message passing layer.

    Parameters
    ----------
    aggr_func : Literal["mean", "sum"], default="sum"
        Aggregation method (Inter-neighborhood).
    update_func : Literal["relu", "sigmoid", "tanh", None], default="sigmoid"
        Update method to apply to merged message.
    """

    def __init__(
        self,
        aggr_func: Literal["mean", "sum"] = "sum",
        update_func: Literal["relu", "sigmoid", "tanh"] | None = "sigmoid",
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.update_func = update_func

    def update(self, inputs):
        """Update (Step 4).

        Parameters
        ----------
        h : array-like, shape = (n_skeleton_out, out_channels)
            Features on the skeleton out.

        Returns
        -------
        array-like, shape = (n_skeleton_out, out_channels)
            Updated features on the skeleton out.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(inputs)
        if self.update_func == "relu":
            return torch.nn.functional.relu(inputs)
        if self.update_func == "tanh":
            return torch.tanh(inputs)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : list
            A list of messages to merge. Each message has shape [n_skeleton_in, channels] and
            len = n_messages_to_merge.

        """
        if self.aggr_func == "sum":
            x = torch.sum(torch.stack(x), axis=0)
        if self.aggr_func == "mean":
            x = torch.mean(torch.stack(x), axis=0)

        if self.update_func is not None:
            x = self.update(x)
        return x
