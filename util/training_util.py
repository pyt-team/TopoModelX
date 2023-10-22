"""Utilities related to training."""

__all__ = ["accuracy", "normalize_features", "encode_labels"]
import numpy as np
import torch
from scipy.sparse import coo_matrix, diags


def accuracy(output, labels):
    """Compute the accuracy of the model.

    Parameters
    ----------
    output : torch.tensor
        Output of the model.
    labels : torch.tensor
        Labels of the data.

    Returns
    -------
    _ : float
        Accuracy of the model.
    """
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def normalize_features(mx, sparse=False, to_torch=False):
    """Normalize features.

    Parameters
    ----------
    mx : array-like
        Features.
    sparse : bool
        Specifies if the matrix is sparse or not.
    to_torch : bool
        Specifies if the matrix should be converted to a torch tensor.

    Returns
    -------
    _ : scipy.sparse.coo_matrix or torch.tensor
        Normalized matrix.
    """
    rowsum = np.array(mx.sum(1))
    r_inv = np.power(rowsum, -1).flatten()
    r_inv[np.isinf(r_inv)] = 0.0
    r_mat_inv = diags(r_inv)
    mx_to = r_mat_inv.dot(mx)
    mx_to = coo_matrix(mx_to)
    if to_torch:
        if sparse:
            mx_to = mx_to.tocoo()
            indices = torch.LongTensor([mx_to.row.tolist(), mx_to.col.tolist()])
            values = torch.Tensor(mx_to.data)
            return torch.sparse.FloatTensor(indices, values)
        return torch.FloatTensor(mx_to.todense())
    return mx_to


def encode_labels(labels):
    """Encode labels.

    Parameters
    ----------
    labels : array-like
        Labels to encode.

    Returns
    -------
    _ : torch.tensor
        Encoded labels.
    """
    classes = {c: i for i, c in enumerate(np.unique(labels))}
    return torch.LongTensor(list(map(classes.get, labels)))
