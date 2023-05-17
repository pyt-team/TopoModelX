"""Message passing module."""

import numpy as np
from scipy.sparse import coo_matrix
from typing import List

import torch
import torch.nn as nn
from torch import Tensor
from torch.nn.parameter import Parameter


from topomodelx.utils.scatter import scatter

class MessagePassing(torch.nn.Module):
    """MessagePassing.

    This corresponds to Steps 1 & 2 of the 4-step scheme.

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
        ------
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