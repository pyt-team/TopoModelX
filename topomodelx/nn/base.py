"""Base class for higher order message passing on topological domains."""

import torch
from topomodelx.utils import scatter


class _Level(torch.nn.Module):
    def ___init___(self, message_passings, inter_agg, update_func):
        """
        Parameters
        ----------
        message_passings : list
            _MessagePassing objects where messages to be aggregated are in sublists
        inter_agg : string
            Aggregation method.
            (Inter-neighborhood).
        update_func : string
            Update function.
        """
        super(_Level, self).__init__()
        self.message_passings = message_passings
        self.inter_agg = inter_agg
        self.update_func = update_func

    def forward(self, x):
        outputs = []
        for mp in self.message_passings:
            if not isinstance(mp, list):
                outputs.append(mp.forward(x))
            else:
                merge = _Merge(mp, inter_agg="sum", update_func="sigmoid")
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
        inter_agg="sum",
        update_func="sigmoid",
        initialization="xavier_uniform",
    ):
        self.message_passings = message_passings
        self.initialization = initialization
        self.inter_agg = inter_agg
        self.update_func = update_func

    def aggregate_inter_neighborhood(self, all_message_x):
        """Aggregate (Step 3).

        Parameters
        ----------
        all_message_x : array-like, shape=[n_neighborhoods, n_skeleton_out, out_channels]
            Messages on one skeleton (out) per neighborhood.
        Returns
        -------
        _ : array-like, shape=[n_skeleton_out, out_channels]
            Aggregated message on one skeleton (out).
        """
        if self.inter_neighborhood_agg == "sum":
            return torch.sum(all_message_x, axis=0)
        return torch.mean(all_message_x, axis=0)

    def update(self, h):
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
            return torch.functional.sigmoid(h)

    def forward(self, h):
        # Step 1 and 2
        all_message_x = []
        for message_passing in self.neighborhoods:
            all_message_x.append(
                message_passing(h)
            )  # .forward(h): TODO change neighborhood name
        # Step 3
        message_x = self.aggregate_inter_neighborhood(all_message_x)
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
    neighborhood : torch.sparse tensors
        Neighborhood matrix.
    intra_aggr : string
        Aggregation method.
        (Inter-neighborhood).
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        neighborhood,
        intra_aggr="sum",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.neighborhood = neighborhood
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
        r"""Construct message going from cell j to cell i.

        Parameters
        ----------
        x : Tensor
            Embedding of the cell j.
        """
        pass

    def aggregate(self, inputs):
        """Aggregate messages from the neighborhood.

        Intra-neighborhood aggregation.
        """
        return scatter(self.intra_aggr)(inputs)

    def update(self, inputs):
        r"""Update embeddings for each cell."""
        # PYG: def update(self, inputs: Tensor) -> Tensor:
        return inputs

    def get_x_i(self, x):
        """Get embedding x_i on cell i."""
        return x.index_select(-2, self.index_i)

    def get_x_j(self, x):
        """Get embedding x_i on cell j."""
        return x.index_select(-2, self.index_j)

    def forward(self, *args, **kwargs):
        r"""Run the forward pass of the module."""
        # forward(self, *args, **kwargs) -> Any:
        pass
