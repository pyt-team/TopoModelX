"""Test the Bscn layer."""

import torch

from topomodelx.nn.simplicial.bScNet_layer import BlockNet
from topomodelx.nn.simplicial.bScNet_layer import testData
import torch_geometric
import torch_geometric.transforms as T


class TestBSCLayer:
    """Test the BSC layer."""

    def test_forward(self):
        """Test the forward pass of the BSC layer."""
        # channels = 5
        # n_nodes = 10
        # n_edges = 20
        # incidence_1 = torch.randint(0, 2, (n_nodes, n_edges)).float()
        # adjacency_0 = torch.randint(0, 2, (n_nodes, n_nodes)).float()
        # x_0 = torch.randn(n_nodes, channels)
        
        dataset = torch_geometric.datasets.Planetoid(
            root='tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
        data=data[0]

        testData,num_features, num_classes, boundary_matrics = testData(data, dataset.name,
                            data.x.size(1), dataset.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bscn = BlockNet(testData, num_features, num_classes,
                        boundary_matrics), data.to(device)

        emb=bscn.g_encode(data)
        #hsn = BSCLayer(channels)
        #output = hsn.forward(x_0, incidence_1, adjacency_0)
        print(emb)
        #assert output.shape == (n_nodes, channels)

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        #channels = 5

        #hsn = BSCLayer(channels)
        #hsn.reset_parameters()
        dataset = torch_geometric.datasets.Planetoid(
            root='tmp/Cora', name='Cora', transform=T.NormalizeFeatures())
        data = data[0]

        testData, num_features, num_classes, boundary_matrics = testData(data, dataset.name,
                                                                         data.x.size(1), dataset.num_classes)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        bscn = BlockNet(testData, num_features, num_classes,
                        boundary_matrics), data.to(device)
        bscn.reset_parameters()
        
        for module in bscn.modules():
            if isinstance(module, torch.nn.Conv2d):
                torch.testing.assert_allclose(
                    module.weight, torch.zeros_like(module.weight)
                )
                torch.testing.assert_allclose(
                    module.bias, torch.zeros_like(module.bias)
                )

        # for module in hsn.modules():
        #     if isinstance(module, torch.nn.Conv2d):
        #         torch.testing.assert_allclose(
        #             module.weight, torch.zeros_like(module.weight)
        #         )
        #         torch.testing.assert_allclose(
        #             module.bias, torch.zeros_like(module.bias)
        #         )
