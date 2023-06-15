import torch

from topomodelx.nn.combinatorial.hmc_layer import HMCLayer


class TestHMCLayer:
    """Unit tests for the HMCLayer class."""

    def test_forward(self):
        """Test the forward method of HMCLayer."""
        in_channels = [1, 2, 3]
        intermediate_channels = [3, 4, 5]
        out_channels = [7, 8, 9]
        hmc_layer = HMCLayer(in_channels, intermediate_channels, out_channels, 0.1)
        adjacency_0 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        adjacency_1 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [1, 2, 0, 2, 0, 1]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        coadjacency_2 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0], [0]]),
            values=torch.tensor([1]),
            size=(1, 1),
            dtype=torch.float,
        )
        incidence_1 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 0, 1, 1, 2, 2], [0, 2, 0, 1, 1, 2]]),
            values=torch.tensor([1, 1, 1, 1, 1, 1]),
            size=(3, 3),
            dtype=torch.float,
        )
        incidence_2 = torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1, 2], [0, 0, 0]]),
            values=torch.tensor([1, 1, 1]),
            size=(3, 1),
            dtype=torch.float,
        )
        x_0 = torch.randn(3, in_channels[0])
        x_1 = torch.randn(3, in_channels[1])
        x_2 = torch.randn(1, in_channels[2])

        x_0_out, x_1_out, x_2_out = hmc_layer.forward(x_0, x_1, x_2, adjacency_0, adjacency_1,
                                                      coadjacency_2, incidence_1, incidence_2)
        assert x_0_out.shape == (3, out_channels[0])
        assert x_1_out.shape == (3, out_channels[1])
        assert x_2_out.shape == (1, out_channels[2])
