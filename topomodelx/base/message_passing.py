"""Message passing module."""
import math
from typing import Literal

import torch

from topomodelx.utils.scatter import scatter


class MessagePassing(torch.nn.Module):
    """Define message passing.

    This class defines message passing through a single neighborhood N,
    by decomposing it into 2 steps:

    1. 🟥 Create messages going from source cells to target cells through N.
    2. 🟧 Aggregate messages coming from different sources cells onto each target cell.

    This class should not be instantiated directly, but rather inherited
    through subclasses that effectively define a message passing function.

    This class does not have trainable weights, but its subclasses should
    define these weights.

    Parameters
    ----------
    aggr_func : Literal["sum", "mean", "add"], default="sum"
        Aggregation function to use.
    att : bool, default=False
        Whether to use attention.
    initialization : Literal["uniform", "xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Initialization method for the weights of the layer.
    initialization_gain : float, default=1.414
        Gain for the weight initialization.

    References
    ----------
    .. [1] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub.
        Topological deep learning: going beyond graph data (2023).
        https://arxiv.org/abs/2206.00606.

    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        aggr_func: Literal["sum", "mean", "add"] = "sum",
        att: bool = False,
        initialization: Literal[
            "uniform", "xavier_uniform", "xavier_normal"
        ] = "xavier_uniform",
        initialization_gain: float = 1.414,
    ) -> None:
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        self.initialization_gain = initialization_gain

    def reset_parameters(self):
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of MessagePassing that have trainable weights.
        """
        match self.initialization:
            case "uniform":
                if self.weight is not None:
                    stdv = 1.0 / math.sqrt(self.weight.size(1))
                    self.weight.data.uniform_(-stdv, stdv)
                if self.att:
                    stdv = 1.0 / math.sqrt(self.att_weight.size(1))
                    self.att_weight.data.uniform_(-stdv, stdv)
            case "xavier_uniform":
                if self.weight is not None:
                    torch.nn.init.xavier_uniform_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_uniform_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case "xavier_normal":
                if self.weight is not None:
                    torch.nn.init.xavier_normal_(
                        self.weight, gain=self.initialization_gain
                    )
                if self.att:
                    torch.nn.init.xavier_normal_(
                        self.att_weight.view(-1, 1), gain=self.initialization_gain
                    )
            case _:
                raise ValueError(
                    f"Initialization {self.initialization} not recognized."
                )

    def message(self, x_source, x_target=None):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the message method in order to replace it with their own message mechanism.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_source_cells, in_channels)
            Messages on source cells.
        """
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages.

        This provides a default attention function to the message-passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        The implementation follows [1]_.

        Parameters
        ----------
        x_source : torch.Tensor, shape = (n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape = (n_target_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        torch.Tensor, shape = (n_messages, 1)
            Attention weights: one scalar per message between a source and a target cell.
        """
        x_source_per_message = x_source[self.source_index_j]
        x_target_per_message = (
            x_source[self.target_index_i]
            if x_target is None
            else x_target[self.target_index_i]
        )

        x_source_target_per_message = torch.cat(
            [x_source_per_message, x_target_per_message], dim=1
        )

        return torch.nn.functional.elu(
            torch.matmul(x_source_target_per_message, self.att_weight)
        )

    def aggregate(self, x_message):
        """Aggregate messages on each target cell.

        A target cell receives messages from several source cells.
        This function aggregates these messages into a single output
        feature per target cell.

        🟧 This function corresponds to the within-neighborhood aggregation
        defined in [1]_ and [2]_.

        Parameters
        ----------
        x_message : torch.Tensor, shape = (..., n_messages, out_channels)
            Features associated with each message.
            One message is sent from a source cell to a target cell.

        Returns
        -------
        Tensor, shape = (...,  n_target_cells, out_channels)
            Output features on target cells.
            Each target cell aggregates messages from several source cells.
            Assumes that all target cells have the same rank s.
        """
        aggr = scatter(self.aggr_func)
        return aggr(x_message, self.target_index_i, 0)

    def forward(self, x_source, neighborhood, x_target=None):
        r"""Forward pass.

        This implements message passing for a given neighborhood:

        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells with input features `x_target`.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        The message passing is decomposed into two steps:

        1. 🟥 Message: A message :math:`m_{y \rightarrow x}^{\left(r \rightarrow s\right)}`
        travels from a source cell :math:`y` of rank r to a target cell :math:`x` of rank s
        through a neighborhood of :math:`x`, denoted :math:`\mathcal{N} (x)`,
        via the message function :math:`M_\mathcal{N}`:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                = M_{\mathcal{N}}\left(\mathbf{h}_x^{(s)}, \mathbf{h}_y^{(r)}, \Theta \right),

        where:

        - :math:`\mathbf{h}_y^{(r)}` are input features on the source cells, called `x_source`,
        - :math:`\mathbf{h}_x^{(s)}` are input features on the target cells, called `x_target`,
        - :math:`\Theta` are optional parameters (weights) of the message passing function.

        Optionally, attention can be applied to the message, such that:

        .. math::
            m_{y \rightarrow x}^{\left(r \rightarrow s\right)}
                \leftarrow att(\mathbf{h}_y^{(r)}, \mathbf{h}_x^{(s)}) . m_{y \rightarrow x}^{\left(r \rightarrow s\right)}

        2. 🟧 Aggregation: Messages are aggregated across source cells :math:`y` belonging to the
        neighborhood :math:`\mathcal{N}(x)`:

        .. math::
            m_x^{\left(r \rightarrow s\right)}
                = \text{AGG}_{y \in \mathcal{N}(x)} m_{y \rightarrow x}^{\left(r\rightarrow s\right)},

        resulting in the within-neighborhood aggregated message :math:`m_x^{\left(r \rightarrow s\right)}`.

        Details can be found in [1]_ and [2]_.

        Parameters
        ----------
        x_source : Tensor, shape = (..., n_source_cells, in_channels)
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape = (n_target_cells, n_source_cells)
            Neighborhood matrix.
        x_target : Tensor, shape = (..., n_target_cells, in_channels)
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        torch.Tensor, shape = (..., n_target_cells, out_channels)
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        neighborhood_values = neighborhood.values()

        x_message = self.message(x_source=x_source, x_target=x_target)
        x_message = x_message.index_select(-2, self.source_index_j)

        if self.att:
            attention_values = self.attention(x_source=x_source, x_target=x_target)
            neighborhood_values = torch.multiply(neighborhood_values, attention_values)

        x_message = neighborhood_values.view(-1, 1) * x_message
        return self.aggregate(x_message)
