"""Test the UniGCNII class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.unigcnii import UniGCNII


class TestUniGCNII:
    """Test the UniGCNII."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        incidence = torch.from_numpy(np.random.rand(2, 2)).to_sparse()
        incidence = incidence.float().to(device)
        model = UniGCNII(num_classes=1, in_features=2, num_layers=2).to(device)

        x_0 = torch.rand(2, 2)

        x_0 = torch.tensor(x_0).float().to(device)

        y1 = model(x_0, incidence)

        assert y1.shape == torch.Size([2, 1])
