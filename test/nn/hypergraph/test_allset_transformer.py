"""Test the AllSetTransformer layer."""

import numpy as np
import torch

from topomodelx.nn.hypergraph.allset_transformer import AllSetTransformer


class TestAllSetTransfomer:
    """Test AllSet."""

    def test_fowared(self):
        """Test forward method."""
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model = AllSetTransformer(
            in_channels=2,
            hidden_channels=2,
            heads=1,
            out_channels=1,
            n_layers=2,
            mlp_num_layers=1,
        )

        x_0 = torch.rand(2, 2)
        incidence_1 = torch.from_numpy(np.random.rand(2, 2)).to_sparse()

        x_0 = torch.tensor(x_0).float().to(device)
        incidence_1 = incidence_1.float().to(device)

        y = model(x_0, incidence_1)
        assert y.shape == torch.Size([])
