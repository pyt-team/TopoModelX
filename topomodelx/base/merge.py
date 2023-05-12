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
        message_passings,
        inter_aggr="sum",
    ):
        super().__init__()
        self.message_passings = message_passings
        self.inter_aggr = inter_aggr

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
            return torch.sum(inputs, axis=0)
        return torch.mean(inputs, axis=0)

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
        return torch.functional.sigmoid(inputs)

    def forward(self, h):
        # Step 1 and 2
        inputs = []
        for message_passing in self.message_passings:
            inputs.append(
                message_passing(h)
            )  # .forward(h): TODO change neighborhood name
        # Step 3
        message_x = self.aggregate(inputs)
        # Step 4
        output_feature_x = self.update(message_x)
        return output_feature_x
