"""Merge module."""

import torch


class _Merge(torch.nn.Module):
    """Message passing layer.

    Parameters
    ----------
    in_channels : int
         Dimension of input features.
    out_channels : int
        Dimension of output features.
    message_passings : list of _MessagePassing objects  # MessagePassingConv objects
        List of (step1, Step2) per neighborhood.
    """

    def __init__(
        self,
        inter_aggr="sum",
        update_on_merge="sigmoid",
    ):
        super().__init__()
        self.inter_aggr = inter_aggr
        self.update_on_merge = update_on_merge

    def aggregate(self, inputs):
        """Aggregate (Step 3).

        Parameters
        ----------
        inputs : array-like, shape=[n_neighborhoods, n_skeleton_out, out_channels]
            Messages on one skeleton (out) per neighborhood.

        Returns
        -------
        _ : array-like, shape=[n_skeleton_out, out_channels]
            Aggregated message on one skeleton (out).
        """
        if self.inter_aggr == "sum":
            return torch.sum(torch.stack(inputs), axis=0)
        return torch.mean(torch.stack(inputs), axis=0)

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
        if self.update_on_merge == "sigmoid":
            return torch.nn.functional.sigmoid(inputs)

    def forward(self, x):
        """Forward pass.

        Parameters
        ----------
        x : list
            len = n_messages_to_merge
            Each message has shape [n_skeleton_in, channels]
        """
        merged_x = self.aggregate(x)
        output_x = self.update(merged_x)
        return output_x
