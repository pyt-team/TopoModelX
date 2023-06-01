"""Message passing module."""

<<<<<<< HEAD
import numpy as np
from scipy.sparse import coo_matrix
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter
=======

import torch
>>>>>>> 03d82bca4fb5c1a611c2a0d97abb7c0ff6cb594d


from topomodelx.utils.scatter import scatter

class MessagePassing(torch.nn.Module):
    """MessagePassing.

    This class defines message passing through a single neighborhood N,
    by decomposing it into 2 steps:

    1. 🟥 Create messages going from source cells to target cells through N.
    2. 🟧 Aggregate messages coming from different sources cells onto each target cell.

    This class should not be instantiated directly, but rather inherited
    through subclasses that effectively define a message passing function.

    This class does not have trainable weights, but its subclasses should
    define these weights.

<<<<<<< HEAD
    """

    def __init__(self, aggregate="sum"):
        super().__init__()
        self.agg = scatter(aggregate)  # Initialize the aggregation function

    def forward(
        self, x: Tensor, a, aggregate_sign=True, aggregate_value=True
    ) -> Tensor:
        """
        Perform the forward pass of the message passing module.
    
        Parameters
        ----------
        x : Tensor
            Input features for each i-cell in the higher order domain.
        a : Tensor or a list of tensors
            Neighborhood matrix representing the connections between i-targert cells and j- source cells.
             One may think about this as cochain map a: C^j -> C^i, i.e. a map that 
             sends signal supported on the j cells to a signal supported on i signal.
        aggregate_sign : bool, optional
            Flag indicating whether to aggregate messages based on the sign of the neighborhood matrix values.
            Default is True.
        aggregate_value : bool, optional
            Flag indicating whether to aggregate messages based on the values of the neighborhood matrix.
            Default is True.
    
        Returns
        -------
        Tensor
            Updated embeddings for each i-cell in the topological domain.    
    
        """      
        
        
        return self.propagate(x, a, aggregate_sign, aggregate_value)

    def _propagate(self, x, a, aggregate_sign=True, aggregate_value=True) -> Tensor:
        """
        Perform message propagation in the topological domain.
    
        Parameters
        ----------
        x : Tensor
            Input features for each i-cell in the topological domain.
        a : Tensor
            Neighborhood matrix representing the connections between i-targert cells and j- source cells.
             One may think about this as cochain map a: C^j -> C^i, i.e. a map that 
             sends signal supported on the j cells to a signal supported on i signal.
        aggregate_sign : bool, optional
            Flag indicating whether to aggregate messages based on the sign of the neighborhood matrix values.
            Default is True.
        aggregate_value : bool, optional
            Flag indicating whether to aggregate messages based on the values of the neighborhood matrix.
            Default is True.
    
        Returns
        -------
        Tensor
            Updated embeddings for each i-cell in the topological domain.
    
        Notes
        -----
        This method performs message passing in the topological domain. 
        It takes input features `x` and a neighborhood matrix `a`,
        and computes messages from source cells to target cells based on the neighborhood matrix.
        The messages are then aggregated based on the target indices defined by the neighborhood matrix,
        and the aggregated messages are used to update the embeddings of the target cells.
    
        The `aggregate_sign` and `aggregate_value` flags control how the messages are aggregated.
        If both flags are True, the messages are multiplied by the values of the neighborhood matrix.
        If only `aggregate_sign` is True, the messages are multiplied by the sign of the
        neighborhood matrix values. If only `aggregate_value` is True, the
        messages are multiplied by the absolute values of the neighborhood matrix.
    
        The method returns the updated embeddings for i-cell in the topological domain.
    
        
    """        
        assert isinstance(x, Tensor) # x must be a torch tensor

        if isinstance(a, (np.ndarray,np.matrix,coo_matrix)):
            a = coo_matrix(a)
            
            self.index_i = torch.from_numpy(a.row.astype(np.int64)).to(torch.long) # Store the target indices for later use in aggregation
            self.index_j = torch.from_numpy(a.col.astype(np.int64)).to(torch.long) # Store the source indices for later use in message passing
            values = torch.from_numpy(a.data.astype(np.float64)).to(torch.float)
        elif isinstance(a,Tensor):    
            target, source = a.coalesce().indices()  # Get the indices of target and source cells
            self.index_i = target  # Store the target indices for later use in aggregation
            self.index_j = source  # Store the source indices for later use in message passing
            values = a.coalesce().values()
        else:
            TypeError("input tensor x must be a torch tonsor, numpy matrix or a coo matrix")
        assert len(a.shape) == 2
        # each source cell sends messages
        # to its "neighbors" according to Nj
        # defined the operator a.
        messages = self.message(x)  # Compute messages from source cells
        if aggregate_sign and aggregate_value:
            #values = a.coalesce().values()  # Get the values of the neighborhood matrix
            messages = torch.mul(values.unsqueeze(-1), messages)  # Multiply values with messages

        elif aggregate_sign and not aggregate_value:
            sign = torch.sign(values)  # Get the sign of the neighborhood matrix values
            messages = torch.mul(sign.unsqueeze(-1), messages)  # Multiply sign with messages

        elif aggregate_value and not aggregate_sign:
            #values = a.coalesce().values()  # Get the values of the neighborhood matrix
            messages = torch.mul(torch.abs(values.unsqueeze(-1)), messages)  # Multiply absolute values with messages

        # each target cell aggregates
        # messages from its neighbors according to Ni
        # defined via the operator a
        embeddings = self.aggregate(messages)  # Aggregate messages from source cells based on target indices

        output = self.update(embeddings)  # Update target cell embeddings

        return output
    
    def propagate(self,x: Tensor , a, aggregate_sign=True, aggregate_value=True ) :
        if isinstance(a,list): 
            for i in a:
                if not isinstance(i, (Tensor,coo_matrix,np.matrix,np.ndarray)):
                    TypeError("each element in the list a must be Tensor,coo_matrix, np.ndarray or np.matrix ")
            return [self.propagate(x,op) for op in a]
        elif  isinstance(a, (Tensor,coo_matrix,np.matrix,np.ndarray)):
            return self._propagate(x, a)
        else:
            TypeError(" input 'a' must be a list of 2d arrays or a 2d array")
            
    def message(self, x: Tensor) -> Tensor:
        return self.get_j(x)  # Extract features from source cells

    def aggregate(self, messages: Tensor):
        return self.agg(messages, self.index_i, 0)  # Perform aggregation of messages based on target indices

    def update(self, embeddings) -> Tensor:
        return embeddings  # Return updated target cell embeddings

    def get_i(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_i)  # Extract features from target cells

    def get_j(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_j)  # Extract features from source cells
=======
    Parameters
    ----------
    aggr_func : string
        Aggregation function to use.
    att : bool
        Whether to use attention.
    initialization : string
        Initialization method for the weights of the layer.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

    .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
        Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
        (2023) https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        aggr_func="sum",
        att=False,
        initialization="xavier_uniform",
    ):
        super().__init__()
        self.aggr_func = aggr_func
        self.att = att
        self.initialization = initialization
        assert initialization in ["xavier_uniform", "xavier_normal"]
        assert aggr_func in ["sum", "mean", "add"]

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)
            if self.att:
                torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight, gain=gain)
            if self.att:
                torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def message(self, x_source, x_target=None):
        """Construct message from source cells to target cells.

        🟥 This provides a default message function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the message method in order to replace it with their own message mechanism.

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : Tensor, shape=[..., n_target_cells, in_channels]
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        _ : Tensor, shape=[..., n_source_cells, in_channels]
            Messages on source cells.
        """
        return x_source

    def attention(self, x_source, x_target=None):
        """Compute attention weights for messages.

        This provides a default attention function to the message passing scheme.

        Alternatively, users can subclass MessagePassing and overwrite
        the attention method in order to replace it with their own attention mechanism.

        Details in [H23]_, Definition of "Attention Higher-Order Message Passing".

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        x_target : torch.Tensor, shape=[n_target_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.

        Returns
        -------
        _ : torch.Tensor, shape = [n_messages, 1]
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
        defined in [H23]_ and [PSHM23]_.

        Parameters
        ----------
        x_messages : Tensor, shape=[..., n_messages, out_channels]
            Features associated with each message.
            One message is sent from a source cell to a target cell.

        Returns
        -------
        _ : Tensor, shape=[...,  n_target_cells, out_channels]
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

        Details in [H23]_ and [PSHM23]_ "The Steps of Message Passing".

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.
        x_target : Tensor, shape=[..., n_target_cells, in_channels]
            Input features on target cells.
            Assumes that all target cells have the same rank s.
            Optional. If not provided, x_target is assumed to be x_source,
            i.e. source cells send messages to themselves.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
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
>>>>>>> 03d82bca4fb5c1a611c2a0d97abb7c0ff6cb594d
