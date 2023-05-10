"""Base classes for higher order message passing on topological domains."""

import torch

from topomodelx.utils import scatter


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


class _MessagePassing(torch.nn.Module):
    """_MessagePassing.

    This corresponds to Steps 1 & 2 of the 4-step scheme.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    intra_aggr : string
        Aggregation method.
        (Inter-neighborhood).
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        intra_aggr="sum",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.intra_aggr = intra_aggr

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        pass

    def propagate(self, x, neighborhood):
        """Start propagating messages."""
        message = self.message(x, neighborhood)
        aggregated_message = self.aggregate(message)
        output = self.update(aggregated_message)

        return output

    def message(self, x, neighborhood):
        r"""Construct message from feature x on source/sender cell.

        Note that this is different from the convention
        in pytorch-geometry which uses x as the features
        that are going to be updated, i.e. on the receiver
        cells.

        Parameters
        ----------
        x : Tensor
            Features on the source cells, that is: the cells
            sending the messages.
        """
        pass

    def aggregate(self, inputs):
        """Aggregate messages from the neighborhood.

        Intra-neighborhood aggregation.
        """
        return scatter(self.intra_aggr)(inputs)

    def update(self, inputs):
        r"""Update embeddings for each cell."""
        return inputs

    def forward(self, *args, **kwargs):
        r"""Run the forward pass of the module."""
        pass
