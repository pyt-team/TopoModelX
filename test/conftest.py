"""Pytest configuration for the TopoModelX test suite."""

import torch

# Explicitly opt out of sparse tensor invariant checks.
torch.sparse.check_sparse_tensor_invariants.disable()
