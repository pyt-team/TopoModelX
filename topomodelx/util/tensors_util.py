"""Utilities related to tensors."""

__all__ = ["sp_softmax", "sp_matmul", "sparse_eye", "batch_mm", "coo_2_torch_tensor"]
import numpy as np
import torch


def sp_softmax(indices, values, N, dim=0):
    """Compute sparse softmax.

    Parameters
    ----------
    indices : torch.tensor, shape=[2, K]
        K is the number of non-zero elements dense matrix corresoponds to the sparse array.
    values : torch.tensor
        Tensor of size K.
    N : int
    dim : int, optional
        Integer that is either 0 or 1, and determines where the sum in the input
        matrix occurs (on source or target simplices/cells).
        The default is 0.

    Returns
    -------
    softmax_v : torch tensor
    """
    if dim == 0:  # sum is on source simplices/cells
        _, ind = indices
    elif dim == 1:  # sum is on target simplices/cells
        ind, _ = indices
    else:
        raise Exception("Input dimension must be 0 o 1.")
    v_max = values.max()
    exp_v = torch.exp(values - v_max)
    exp_sum = torch.zeros(N, 1)

    exp_sum.scatter_add_(0, ind.unsqueeze(1), exp_v.float())
    exp_sum += 1e-10
    softmax_v = exp_v / exp_sum[ind]
    return softmax_v


def sp_matmul(indices, values, mat, output_size, dim=0):
    """Compute sparse matrix multiplication.

    This function performs simple sparse matrix multiplication between a sparse
    array represented by indices and values, and a dense array represeted by mat

    Parameters
    ----------
    indices : torch.tensor, shape=[2, K]
        K is the number of non-zero elements dense matrix corresoponds to the sparse array.
    values : torch.tensor
        Tensor of size K.
    mat : torch dense matrix
    output_size : tuple of length 2
        Shape of the output matrix.

    Returns
    -------
    out : torch.tensor, shape=output_size
        Result of the sparse matrix multiplication.
    """
    source, target = indices
    out = torch.zeros(output_size)
    if dim == 0:
        out.scatter_add_(0, target.expand(mat.size(1), -1).t(), values * mat[source])
    elif dim == 1:
        out.scatter_add_(0, source.expand(mat.size(1), -1).t(), values * mat[target])
    else:
        raise

    return out


def sparse_eye(size):
    """Return the identity matrix as a sparse matrix.

    See: https://www.programcreek.com/python/example/101243/torch.sparse

    Parameters
    ----------
    size : int
        Size of the identity matrix.

    Returns
    -------
    _ : array-like
        Identity matrix of size `size`.
    """
    indices = torch.arange(0, size).long().unsqueeze(0).expand(2, size)
    values = torch.tensor(1.0).expand(size)
    cls = getattr(torch.sparse, values.type().split(".")[-1])
    return cls(indices, values, torch.Size([size, size]))


def coo_2_torch_tensor(sparse_mx, sparse=True):
    """Convert a scipy matrix to a torch tensor.

    Parameters
    ----------
    sparse_mx : scipy matrix
        Matrix to convert.
    sparse: bool
        Specifies if the matrix is sparse or not.

    Returns
    -------
    _ : torch.tensor
        Converted matrix.
    """
    if sparse:
        sparse_mx = sparse_mx.tocoo().astype(np.float32)
        indices = torch.from_numpy(
            np.vstack((sparse_mx.row, sparse_mx.col)).astype(np.int64)
        )
        values = torch.from_numpy(sparse_mx.data)
        shape = torch.Size(sparse_mx.shape)
        return torch.sparse.FloatTensor(indices, values, shape)
    return torch.FloatTensor(sparse_mx.todense())


def batch_mm(matrix, matrix_batch):
    """Compute a batched matrix-matrix product.

    See: https://github.com/pytorch/pytorch/issues/14489

    Parameters
    ----------
    matrix: Sparse or dense matrix, shape=[m, n]
        Matrix.
    matrix_batch: Batched dense matrices, shape=[b, n, k]
        Batch of matrices.

    Returns
    -------
    _ : array-like
        The batched matrix-matrix product, shape (m, n) x (b, n, k) = (b, m, k).
    """
    batch_size = matrix_batch.shape[0]
    # Stack the vector batch into columns. (b, n, k) -> (n, b, k) -> (n, b*k)
    vectors = matrix_batch.transpose(0, 1).reshape(matrix.shape[1], -1)

    # A matrix-matrix product is a batched matrix-vector product of the columns.
    # And then reverse the reshaping. (m, n) x (n, b*k) = (m, b*k) -> (m, b, k) -> (b, m, k)
    return matrix.mm(vectors).reshape(matrix.shape[0], batch_size, -1).transpose(1, 0)


def sparse_mx_to_torch_sparse_tensor(sparse_mx):
    """Convert a scipy sparse matrix to a torch sparse tensor.

    Parameters
    ----------
    sparse_mx : scipy sparse matrix
        Matrix to convert.

    Returns
    -------
    _ : torch.sparse.FloatTensor
        Converted matrix.
    """
    return coo_2_torch_tensor(sparse_mx, sparse=True)
