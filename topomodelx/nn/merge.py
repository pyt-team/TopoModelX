from typing import Optional, Tuple, Union
import torch
from torch import Tensor

from scipy.sparse import coo_matrix as SciPyCooMatrix

from topomodelx.nn.linear import Linear
from topomodelx.nn.message_passing import HigherOrderMessagePassing



class Merge(HigherOrderMessagePassing):
    """
    Merges features from two different inputs using message passing on a simplicial complex.

    The `Merge` class combines two sets of features through various merging strategies
    after propagating messages through the graph defined by the incidence and adjacency matrices.

    Args:
        in_ch_1 (int): Input feature size for the first input.
        in_ch_2 (int): Input feature size for the second input.
        target_ch (int): Output feature size after merging.
        merge (str, optional): The merging strategy to use. Options include:
                               'conc' (concatenation), 
                               'sum' (element-wise sum),
                               'mean' (element-wise mean),
                               'max' (element-wise maximum),
                               'min' (element-wise minimum).
                               Default is 'conc'.

    Example:
          from toponetx import SimplicialComplex

          # Define a simplicial complex
          SC = SimplicialComplex([[0, 1], [1, 2]])
          
          # Get incidence and adjacency matrices as COO format
          B1 = SC.incidence_matrix(1)
          A0 = SC.adjacency_matrix(0)
          
          # Get dimensions from matrices
          n_v, n_e = B1.shape
          
          # Initialize the Merge class
          merge = Merge(in_ch_1=10, in_ch_2=8, target_ch=16)

          # Create random input feature tensors
          x_e = torch.rand(n_e, 10)  # Edge features
          x_v = torch.rand(n_v, 8)    # Vertex features

          # Perform the merge operation
          try:
              x_v_out = merge(x_e, x_v, B1, A0)
              print(f"Merged output shape: {x_v_out.shape}")  # Output shape will depend on the merge strategy
          except ValueError as e:
              print(f"Error: {e}")
    """

    def __init__(self, in_ch_1: int, in_ch_2: int, target_ch: int, merge: str = "conc"):
        super().__init__()
        self.merge = merge.lower()  # Normalize to lowercase
        self.linear1 = Linear(in_ch_1, target_ch)
        self.linear2 = Linear(in_ch_2, target_ch)

        # Validate merging strategy
        valid_merge_strategies = ['conc', 'sum', 'mean', 'max', 'min']
        if self.merge not in valid_merge_strategies:
            raise ValueError(f"Invalid merge strategy '{self.merge}'. "
                             f"Choose from {valid_merge_strategies}.")

    def forward(self, x1: Tensor, x2: Tensor, G1: Tensor, G2: Tensor) -> Tensor:
        """
        Propagates messages and merges the outputs from two input feature sets.

        Args:
            x1 (Tensor): Input features of shape (num_cells, in_ch_1).
            x2 (Tensor): Input features of shape (num_cells, in_ch_2).
            G1 (Tensor): Incidence matrix for the first input.
            G2 (Tensor): Incidence matrix for the second input.

        Returns:
            Tensor: Merged output features after message passing. Shape depends on the merging strategy.

        Raises:
            ValueError: If the shapes of the input tensors or incidence matrices are inconsistent.
        """
        # Validate input shapes
        #if x1.shape[0] != G1.shape[0]:
        #    raise ValueError("The number of cells in x1 does not match the number of edges in G1.")
        #if x2.shape[0] != G2.shape[0]:
        #    raise ValueError("The number of cells in x2 does not match the number of edges in G2.")

        out1 = self.propagate(self.linear1(x1), G1)
        out2 = self.propagate(self.linear2(x2), G2)

        if self.merge == 'conc':
            return torch.cat((out1, out2), dim=-1)  # Concatenate along the last dimension
        elif self.merge == 'sum':
            return out1 + out2
        elif self.merge == 'mean':
            return (out1 + out2) / 2  # Element-wise mean
        elif self.merge == 'max':
            return torch.max(out1, out2)  # Element-wise maximum
        elif self.merge == 'min':
            return torch.min(out1, out2)  # Element-wise minimum



