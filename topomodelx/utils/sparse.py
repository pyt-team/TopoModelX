"""Utils for more efficient sparse matrix casting to torch."""

import numpy as np
import torch
from scipy import sparse


def from_sparse(data: sparse.spmatrix | sparse.sparray) -> torch.Tensor:
    """Convert sparse input data directly to torch sparse coo format.

    Parameters
    ----------
    data : scipy.sparse.spmatrix or scipy.sparse.sparray
        Input n_dimensional data.

    Returns
    -------
    torch.sparse_coo, same shape as data
        input data converted to tensor.
    """
    # cast from csc_matrix to coo format for compatibility
    coo = data.tocoo()

    values = torch.FloatTensor(coo.data)
    indices = torch.LongTensor(np.vstack((coo.row, coo.col)))

    return torch.sparse_coo_tensor(
        indices, values, coo.shape, check_invariants=False
    ).coalesce()
