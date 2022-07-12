__all__ = [
    "get_normalized_2d_operators",
    "_compute_B1_normalized",
    "_compute_B1T_normalized",
    "_compute_B2_normalized",
    "_compute_B2T_normalized",
    "_compute_D1",
    "_compute_D2",
    "_compute_D3",
    "_compute_D5",
]


import numpy as np
from numpy import ndarray
from numpy.linalg import pinv
from scipy.linalg import pinv as s_pinv
from scipy.sparse import diags, identity
from scipy.sparse.coo import coo_matrix

"""
@author: Mustafa Hajij

refs :
    [1] Michael T. Schaub, Austin R. Benson,
        Paul Horn, Gabor Lippner,
        Ali Jadbabaie Random walks on
        simplicial complexes and the normalized
        hodge 1-laplacian.
    [2] Eric Bunch, Qian You,
        Glenn Fung, Vikas Singh
        Simplicial 2-Complex Convolutional
        Neural Networks
"""


def get_normalized_2d_operators(B1, B2):
    """
    Args
    ------
        B1 : The boundary B1: C1->C0  of a simplicial complex.
            type: numpy array or scipy coo_matrix
        B2 : The boundary B2: C2->C1  of a simplicial complex.
            type:  numpy array or scipy coo_matrix

    Returns
    -------
        B1 : normalized B1 : C1->C2
             numpy array or scipy coo_matrix
        B1T : normalized B1T : C0->C1
             numpy array or scipy coo_matrix
        B2 : normalized B2 : C2->C1
             numpy array or scipy coo_matrix
        B2T : normalized B2T : C1->C2
             numpy array or scipy coo_matrix

    """
    B1N = _compute_B1_normalized(B1, B2)
    B1TN = _compute_B1T_normalized(B1, B2)
    B2N = _compute_B2_normalized(B2)
    B2TN = _compute_B2T_normalized(B2)
    return B1N, B1TN, B2N, B2TN


def _compute_B1_normalized(B1, B2):
    """
    args:
        B1 : The boundary B1: C1->C0  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
        B2 : The boundary B2: C2->C1  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    return:
            The normalized boundary B1: C1->C0  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, ndarray):
        D1_pinv = pinv(D1)
    elif isinstance(B1, coo_matrix):
        D1_pinv = coo_matrix(pinv(D1.toarray()))
    return D1_pinv @ B1


def _compute_B1T_normalized(B1, B2):
    """
    args:
        B1 : The boundary B1: C1->C0  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
        B2 : The boundary B1: C2->C1  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    return:
            The normalized traponse boundary operator B1T: C0->C1
            of a simplicial complex. This is the same as be regarded as the
            coboundary C0->C1
            type:  numpy array or scipy coo_matrix
    """
    D2 = _compute_D2(B2)
    D1 = _compute_D1(B1, D2)
    if isinstance(B1, ndarray):
        D1_pinv = pinv(D1)
    elif isinstance(B1, coo_matrix):
        D1_pinv = coo_matrix(pinv(D1.toarray()))
    else:
        raise TypeError("input type must be either ndarray or coo_matrix")
    return D2 @ B1.T @ D1_pinv


def _compute_B2_normalized(B2):
    """
    args:

        B2 : The boundary B1: C2->C1  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    return:
            The normalized  boundary operator B2: C2->C1
            of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    """
    D3 = _compute_D3(B2)
    return B2 @ D3


def _compute_B2T_normalized(B2):
    """
    args:
        B2 : The boundary B1: C2->C1  of a simplicial complex.
            type:  numpy array or scipy coo_matrix
    return:
            The normalized traponse boundary operator B2T: C1->C2
            of a simplicial complex. This is the same
            as be regarded as the coboundary C1->C2
            type:  numpy array or scipy coo_matrix
    """
    D5 = _compute_D5(B2)
    if isinstance(B2, ndarray):
        D5_pinv = pinv(D5)
    elif isinstance(B2, coo_matrix):
        D5_pinv = coo_matrix(pinv(D5.toarray()))
    else:
        raise TypeError("input type must be either ndarray or coo_matrix")
    return B2.T @ D5_pinv


def _compute_D1(B1, D2):
    """
    Note:
        use in normalization of B1 and B1T
    """
    if isinstance(B1, coo_matrix):
        rowsum = np.array((abs(B1) @ D2).sum(axis=1)).flatten()
        D1 = 2 * diags(rowsum)
    elif isinstance(B1, ndarray):
        rowsum = (np.abs(B1) @ D2).sum(axis=1)
        D1 = 2 * np.diag(rowsum)
    else:
        raise TypeError("input type must be either ndarray or coo_matrix")
    return D1


def _compute_D2(B2):
    """
    Note:
        use in normalization of B1T
    """
    if isinstance(B2, coo_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        D2 = diags(np.maximum(rowsum, 1))
    elif isinstance(B2, ndarray):
        rowsum = np.abs(B2).sum(axis=1)
        D2 = np.diag(np.maximum(rowsum, 1))
    else:
        raise TypeError("input type must be either ndarray or coo_matrix")
    return D2


def _compute_D3(B2):
    """
    Note:
        use in normalization of B2
    """
    if isinstance(B2, coo_matrix):
        D3 = identity(B2.shape[1]) / 3
    elif isinstance(B2, ndarray):
        D3 = np.identity(B2.shape[1]) / 3
    return D3


def _compute_D5(B2):
    """
    Note:
        use in normalization of B2T
    """
    if isinstance(B2, coo_matrix):
        rowsum = np.array(np.abs(B2).sum(axis=1)).flatten()
        D5 = diags(rowsum)
    elif isinstance(B2, ndarray):
        rowsum = (np.abs(B2)).sum(axis=1)
        D5 = np.diag(rowsum)
    else:
        raise TypeError("input type must be either ndarray or coo_matrix")
    return D5
