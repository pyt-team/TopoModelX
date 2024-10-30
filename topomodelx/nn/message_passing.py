import torch
from topomodelx.scatter.scatter import scatter
from typing import Optional, Tuple
import torch
from torch import Tensor
import scipy.sparse as sp
import numpy as np

__all_=["HigherOrderMessagePassing"]

class HigherOrderMessagePassing(torch.nn.Module):
    """
    A higher-order message-passing layer for simplicial complexes and hypergraphs, as introduced in 
    *Hajij et el. Topological Deep Learning: Going Beyond Graph Data*. This layer generalizes message passing to higher-order 
    structures beyond graphs, such as simplices and hyperedges, enabling richer relational learning in higher-order 
    networks.

    Parameters
    ----------
    aggregate : str, default="sum"
        The aggregation method used to combine messages. Options include "sum", "mean", "max", among others,
        depending on the available operations in the scatter function. The choice of aggregation influences
        how messages are aggregated across simplices or hyperedges.

    Methods
    -------
    forward(x, a, aggregate_sign=True, aggregate_value=True)
        Executes forward pass of higher-order message passing by propagating features on the given structure.
    
    propagate(x, a, aggregate_sign=True, aggregate_value=True)
        The main message-passing logic. Gathers and aggregates messages, based on adjacency and
        feature values, for higher-order relations.

    message(x)
        Computes the initial message for each element in the tensor.

    aggregate(messages)
        Aggregates messages based on the specified indices and aggregation method.

    update(embeddings)
        Final transformation of the aggregated embeddings, if any.

    convert_sparse_to_torch(sparse_matrix)
        Converts a SciPy sparse matrix to a PyTorch sparse tensor.

    Examples
    --------
    >>> from toponetx.simplicial_complex import SimplicialComplex
    >>> SC = SimplicialComplex([[0,1],[1,2]])
    >>> B1 = SC.incidence_matrix(1)
    >>> homp = HigherOrderMessagePassing()
    >>> x_e = torch.rand(2, 10)
    >>> x_v = homp(x_e, B1)
    >>> print(x_v)

    Math
    ----
    In higher-order message passing, each message update for element `i` is defined as:
    
    .. math::
        h_i^{(k+1)} = \sigma \left( \sum_{j \in \mathcal{N}(i)} \alpha_{ij} \phi(h_j^{(k)}, h_i^{(k)}) \right)

    where:
        - :math:`h_i^{(k)}` is the feature vector of element `i` at layer `k`,
        - :math:`\phi` represents a function to compute messages between elements,
        - :math:`\alpha_{ij}` is the weight (e.g., sign or absolute value) for the edge from `j` to `i`,
        - :math:`\sigma` is an activation function or update rule applied after aggregation.

    This class supports different configurations of :math:`\alpha_{ij}` using `aggregate_sign` and `aggregate_value` 
    flags to control sign and magnitude usage in messages.

    Raises
    ------
    ValueError
        If the input adjacency is not a 2D matrix or has unsupported format.
    TypeError
        If the input matrix is neither a PyTorch tensor nor a SciPy sparse matrix.

    """

    def __init__(self, aggregate="sum"):
        super().__init__()
        self.agg = scatter(aggregate)

    def forward(
        self, x: Tensor, a: Tensor, aggregate_sign: bool = True, aggregate_value: bool = True
    ) -> Tensor:
        """
        Executes forward propagation using higher-order message passing.

        Parameters
        ----------
        x : Tensor
            Node or simplex feature tensor of shape `(num_elements, feature_dim)`.
        a : Tensor or sp.spmatrix
            Adjacency matrix in sparse COO format representing connections.
        aggregate_sign : bool, optional
            Whether to consider the sign of adjacency weights in messages.
        aggregate_value : bool, optional
            Whether to consider the magnitude of adjacency weights in messages.

        Returns
        -------
        Tensor
            The output features after higher-order message passing.
        """
        return self.propagate(x, a, aggregate_sign, aggregate_value)

    def propagate(self, x: Tensor, a: Tensor, aggregate_sign = True, aggregate_value = True) -> Tensor:
        # Validate inputs
        if not isinstance(x, Tensor):
            raise TypeError("Input x must be a PyTorch tensor.")
        if not isinstance(a, (Tensor, sp.spmatrix)):
            raise TypeError("Input a must be a PyTorch tensor or a SciPy sparse matrix.")

        # Convert SciPy sparse matrix to COO format if necessary
        if isinstance(a, sp.spmatrix):
            a = self.convert_sparse_to_torch(a)

        # Convert dense tensor to sparse if necessary
        if not a.is_sparse:
            a = a.to_sparse()

        if len(a.shape) != 2:
            raise ValueError("Input a must be a 2D matrix.")

        # Extract indices for sparse tensor
        indices = a.coalesce().indices()
        target, source = indices[0], indices[1]

        self.index_i = target  # Target index
        self.index_j = source  # Source index

        messages = self.message(x)
        if aggregate_sign and aggregate_value:
            values = a.coalesce().values()
            messages = torch.mul(values.unsqueeze(-1), messages)
        elif aggregate_sign and not aggregate_value:
            sign = torch.sign(a.coalesce().values())
            messages = torch.mul(sign.unsqueeze(-1), messages)
        elif aggregate_value and not aggregate_sign:
            values = a.coalesce().values()
            messages = torch.mul(torch.abs(values.unsqueeze(-1)), messages)

        embeddings = self.aggregate(messages)
        output = self.update(embeddings)

        return output

    def convert_sparse_to_torch(self, sparse_matrix: sp.spmatrix) -> Tensor:
        """
        Converts a SciPy sparse matrix to a PyTorch sparse tensor.

        Parameters
        ----------
        sparse_matrix : sp.spmatrix
            The SciPy sparse matrix to convert.

        Returns
        -------
        Tensor
            The PyTorch sparse tensor in COO format.

        Raises
        ------
        ValueError
            If an unsupported sparse matrix format is provided.
        """
        if sp.issparse(sparse_matrix):
            if sparse_matrix.format == 'coo':
                indices = torch.tensor(np.vstack((sparse_matrix.row, sparse_matrix.col)), dtype=torch.long)
                values = torch.tensor(sparse_matrix.data, dtype=torch.float32)
                return torch.sparse_coo_tensor(indices, values, sparse_matrix.shape)
            elif sparse_matrix.format in ['csr', 'csc']:
                coo_matrix = sparse_matrix.tocoo()
                return self.convert_sparse_to_torch(coo_matrix)
            else:
                raise ValueError("Unsupported sparse matrix format. Please use COO, CSR, or CSC.")
        else:
            raise TypeError("Input must be a SciPy sparse matrix.")

    def message(self, x: Tensor) -> Tensor:
        return self.get_j(x)

    def aggregate(self, messages: Tensor) -> Tensor:
        return self.agg(messages, self.index_i, 0)

    def update(self, embeddings: Tensor) -> Tensor:
        return embeddings

    def get_i(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_i)

    def get_j(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_j)
