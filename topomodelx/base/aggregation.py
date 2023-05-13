"""Aggregation module."""

import torch


class Aggregation(torch.nn.Module):
    """Message passing layer.

    Parameters
    ----------
    aggr_func : string
        Aggregation method.
        (Inter-neighborhood).
    update_func : string
        Update method to apply to merged message.
    """

    def __init__(
        self,
        aggr_func="sum",
        update_func="sigmoid",
    ):
        super().__init__()
        self.aggr_func = aggr_func
        self.update_func = update_func

    def update(self, inputs):
        """Update (Step 4).

        Parameters
        ----------
        h : array-like, shape=[n_skleton_out, out_channels]
            Features on the skeleton out.

        Returns
        -------
        _ : array-like, shape=[n_skleton_out, out_channels]
            Updated features on the skeleton out.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(inputs)
        if self.update_func == "relu":
            return torch.nn.functional.relu(inputs)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : list
            len = n_messages_to_merge
            Each message has shape [n_skeleton_in, channels]
        """
        if self.aggr_func == "sum":
            x = torch.sum(torch.stack(x), axis=0)
        x = torch.mean(torch.stack(x), axis=0)
            
        if self.update_func is not None:
            x = self.update(x)
        return x
