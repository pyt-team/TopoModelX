"""Utils for more efficient sparse matrix casting to torch."""

import numpy as np
import scipy
import torch


def from_sparse(data: scipy.sparse._csc.csc_matrix, dense=False):
    """Convert sparse input data directly to torch sparse coo format.

    Parameters
    ----------
    data : scipy.sparse._csc.csc_matrix
        Input n_dimensional data.

    Returns
    -------
    torch.sparse_coo, same shape as data
        input data converted to tensor.
    """
    if not isinstance(data, scipy.sparse._csc.csc_matrix):
        raise ValueError(
            f"Expected Data type scipy.sparse._csc.csc_matrix, found {type(data)}"
        )

    # cast from csc_matrix to coo format for compatibility
    coo = data.tocoo()

    v = torch.FloatTensor(coo.data)
    i = torch.LongTensor(np.vstack((coo.row, coo.col)))

    out = torch.sparse_coo_tensor(i, v, coo.shape)

    if dense:
        return out.to_dense()

    return out
