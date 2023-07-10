"""Utils for row normalization in sparse tensors."""


import torch


def sparse_row_norm(sparse_tensor):
    r"""Normalize a sparse tensor by row dividing each row by its sum.

    Parameters
    ----------
    sparse_tensor : torch.sparse, shape=[n_cells, n_cells]

    Returns
    -------
    _ : torch.sparse, shape=[n_cells, n_cells]
        Normalized by rows sparse tensor.
    """
    row_sum = torch.sparse.sum(sparse_tensor, dim=1)
    values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor._indices(), values, sparse_tensor.shape
    )
    return sparse_tensor.coalesce()
