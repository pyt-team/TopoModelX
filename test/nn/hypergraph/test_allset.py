"""Allset class."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.allset import AllSet


class TestAllSet:
    """Test AllSet."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AllSet(
            in_channels=4,
            hidden_channels=4,
            out_channels=4,
            n_layers=2,
            mlp_num_layers=1,
        )

        x_0 = torch.rand(4, 4)
        incidence_1 = torch.from_numpy(np.random.rand(4, 4)).to_sparse()

        x_0 = torch.tensor(x_0).float().to(device)
        incidence_1 = incidence_1.float().to(device)

        y = model(x_0, incidence_1)
        assert y.shape == torch.Size([])
