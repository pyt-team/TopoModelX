from typing import Optional, Tuple, Union
import torch
from torch import Tensor

from scipy.sparse import coo_matrix as SciPyCooMatrix

from topomodelx.nn.linear import Linear
from topomodelx.nn.message_passing import HigherOrderMessagePassing

__all__ = ["GeneralMerge"]

class GeneralMerge(HigherOrderMessagePassing):
    """
    Merges features from an arbitrary number of input feature tensors using message passing on 
    simplicial complexes, with various merging strategies.

    This class enables combining multiple sets of input features through customizable message 
    passing and aggregation schemes, based on specified incidence or adjacency matrices.

    Args:
        in_channels (List[int]): List of input feature sizes for each feature tensor.
        target_channels (int): Output feature size after merging.
        merge (str, optional): Merging strategy to apply. Options are:
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
                
                # Obtain incidence and adjacency matrices as COO format
        B1 = SC.incidence_matrix(1)
        A0 = SC.adjacency_matrix(0)
                
                # Define dimensions and initialize the class
        n_v, n_e = B1.shape
        merge = GeneralizedMerge(in_channels=[10, 8, 6], target_channels=16)

                # Create random input feature tensors
        x1 = torch.rand(n_e, 10)  # First feature tensor
        x2 = torch.rand(n_v, 8)   # Second feature tensor
        x3 = torch.rand(n_v, 6)   # Third feature tensor


        output = merge([x1, x2, x3], [B1, A0, A0])
        print(f"Merged output shape: {output.shape}")  # Output shape depends on merge strategy
    """

    def __init__(self, in_channels: List[int], target_channels: int, merge: str = "conc"):
        super().__init__()
        self.merge = merge.lower()  # Normalize to lowercase
        self.linears = nn.ModuleList([nn.Linear(in_ch, target_channels) for in_ch in in_channels])

        # Validate merging strategy
        valid_merge_strategies = ['conc', 'sum', 'mean', 'max', 'min']
        if self.merge not in valid_merge_strategies:
            raise ValueError(f"Invalid merge strategy '{self.merge}'. "
                             f"Choose from {valid_merge_strategies}.")

    def forward(self, x_list: List[Tensor], matrix_list: List[Union[Tensor, SciPyCooMatrix]]) -> Tensor:
        """
        Propagates messages and merges outputs from multiple input feature sets.

        Args:
            x_list (List[Tensor]): List of input features, each of shape (num_cells, in_ch_i).
            matrix_list (List[Union[Tensor, SciPyCooMatrix]]): List of incidence or adjacency matrices.

        Returns:
            Tensor: Merged output features after message passing. Shape depends on the merging strategy.

        Raises:
            ValueError: If the number of inputs does not match the number of matrices, or if shapes are inconsistent.
        """
        if len(x_list) != len(matrix_list):
            raise ValueError("The number of feature tensors must match the number of matrices.")

        propagated_features = []
        for i, (x, matrix) in enumerate(zip(x_list, matrix_list)):
            
            # Apply linear transformation and propagate
            transformed_x = self.linears[i](x)
            propagated_features.append(self.propagate(transformed_x, matrix))

        # Apply the chosen merging strategy
        if self.merge == 'conc':
            return torch.cat(propagated_features, dim=-1)  # Concatenate along the last dimension
        elif self.merge == 'sum':
            return torch.stack(propagated_features).sum(dim=0)  # Sum across inputs
        elif self.merge == 'mean':
            return torch.stack(propagated_features).mean(dim=0)  # Mean across inputs
        elif self.merge == 'max':
            return torch.stack(propagated_features).max(dim=0).values  # Max across inputs
        elif self.merge == 'min':
            return torch.stack(propagated_features).min(dim=0).values  # Min across inputs


    def _convert_sparse_to_torch(self, sparse_matrix: SciPyCooMatrix) -> Tensor:
        """
        Convert a SciPy sparse COO matrix to a PyTorch sparse tensor.
        """
        indices = torch.tensor([sparse_matrix.row, sparse_matrix.col], dtype=torch.long)
        values = torch.tensor(sparse_matrix.data, dtype=torch.float32)
        return torch.sparse_coo_tensor(indices, values, sparse_matrix.shape)




