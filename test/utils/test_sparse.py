import numpy as np
import pytest
import torch
from scipy import sparse

from topomodelx.utils.sparse import from_sparse


def test_from_sparse():
    # test numerical matching
    test_matrix = sparse._csc.csc_matrix(np.random.rand(100, 100))
    a = torch.from_numpy(test_matrix.todense()).to_sparse()
    b = from_sparse(test_matrix)

    assert torch.equal(
        a.to_dense().type(torch.float32), b.to_dense().type(torch.float32)
    )

    # test on larger dimension
    test_matrix = sparse.csc_matrix(
        ([3, 4, 5, -3.2], ([0, 1, 1, 200], [2, 0, 2, 500])), shape=(2000, 3000)
    )
    a = torch.from_numpy(test_matrix.todense()).to_sparse()
    b = from_sparse(test_matrix)

    assert torch.equal(
        a.to_dense().type(torch.float32), b.to_dense().type(torch.float32)
    )


def test_fail_on_wrong_type():
    with pytest.raises(ValueError):
        test_matrix = np.random.rand(100, 100)
        res = from_sparse(test_matrix)
