"""Test the UniGCN class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.unigcn import UniGCN


class TestUniGCN:
    """Test the UniGCN."""

    def test_forward(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        incidence = torch.from_numpy(np.random.rand(2, 2)).to_sparse()
        incidence = incidence.float().to(device)
        model = UniGCN(channels_edge=2, channels_node=2, n_layers=2).to(device)

        x_0 = torch.rand(2, 2)

        x_0 = torch.tensor(x_0).float().to(device)

        y1 = model(x_0, incidence)

        assert y1.shape == torch.Size([1])
