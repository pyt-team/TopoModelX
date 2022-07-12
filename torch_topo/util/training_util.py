__all__ = ["accuracy", "normalize_features", "encode_labels"]
import numpy as np
import torch
from scipy.sparse import diags, coo_matrix


def accuracy(output, labels):
    preds = output.max(1)[1]
    correct = preds.eq(labels).sum().item()
    return correct / len(labels)


def normalize_features(mx, sparse=False, to_torch=False):
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
        else:
            return torch.FloatTensor(mx_to.todense())
    else:
        return mx_to


def encode_labels(labels):
    classes = {c: i for i, c in enumerate(np.unique(labels))}
    return torch.LongTensor(list(map(classes.get, labels)))
