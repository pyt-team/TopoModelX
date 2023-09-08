"""Dynamic hypergraph convolutional network (DHGCN) Layer implementation."""
import torch
from torch_cluster import knn_graph, nearest

from topomodelx.base.conv import Conv


class DHGCNLayer(torch.nn.Module):
    """Dynamic Topology Layer of a Dynamic hypergraph convolutional network (DHGCN).

    Dynamic topology, followed by two-step message passing layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    intermediate_channels : int
        Dimension of intermediate features.
    out_channels : int
        Dimension of output features.

    References
    ----------
    .. [Y22] Yin N, Feng F, Luo Z, Zhang X, Wang W, Luo X, Chen C, Hua XS.
        Dynamic hypergraph convolutional network.
        In2022 IEEE 38th International Conference on Data Engineering (ICDE) 2022 May 9 (pp. 1621-1634). IEEE.
        https://ieeexplore.ieee.org/abstract/document/9835240
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
        k_neighbours: int = 3,
        k_centroids: int = 4,
        device: str = "cpu",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.intermediate_channels = intermediate_channels
        self.k_neighbours = k_neighbours
        self.k_centroids = k_centroids

        self.fc_layer = torch.nn.Linear(
            in_channels, intermediate_channels, bias=True, device=device
        )

        self.conv_dt_level0_0_to_1 = Conv(
            in_channels=intermediate_channels,
            out_channels=intermediate_channels,
        )
        self.conv_dt_level0_1_to_0 = Conv(
            in_channels=intermediate_channels, out_channels=out_channels
        )

    @staticmethod
    def kmeans_graph(x, k, flow: str = "source_to_target"):
        r"""K-means algorithm implementation.

        Parameters
        ----------
        x : torch.Tensor, shape=[n_nodes, node_features]
            Input features on the nodes of the simplicial complex.
        k : int
            Number of clusters/centroids
        flow : str
            If this parameter has value "source_to_target", the output will have the shape
            [n_nodes, n_hyperedges = k_centroids].
            If this parameter has value "target_to_source", the output shape will be
            [n_hyperedges = k_centroids, n_nodes].
            It corresponds to the pytorch_cluster's flow parameter is the knn_graph method
            and is defined accordingly.

        Returns
        -------
        hyperedge_index : torch.Tensor, shape=[n_nodes, 2]
            Indices of the on-zero values in the feature matrix of hypergraph
            convolutional network.
            The order of dimensions of the matrix is defined by the value of the flow
            parameter.
        """
        assert flow in ["source_to_target", "target_to_source"]
        device = x.device
        permutation = torch.randperm(x.size(0), device=device)
        centroid_indices = permutation[:k]
        element_indices_by_cluster = [centroid_indices[i : i + 1] for i in range(k)]
        for new_index in permutation[k:]:
            centroids = torch.index_select(x, 0, centroid_indices)

            new_element = x[new_index : new_index + 1]
            batch_x = torch.tensor([0], device=device)
            batch_y = torch.zeros(centroids.size(0), device=device)
            assigned_cluster = nearest(new_element, centroids, batch_x, batch_y)

            element_indices = element_indices_by_cluster[assigned_cluster]

            new_cluster_indices = torch.cat(
                [
                    element_indices,
                    torch.tensor(
                        [new_index], dtype=element_indices.dtype, device=device
                    ),
                ]
            )
            new_features = torch.index_select(x, 0, new_cluster_indices)
            distances = torch.cdist(new_features, new_features)
            mean_distances = torch.stack(
                [x.sum() / (x.size(0) - 1) for x in distances.unbind()]
            )

            min_index = torch.argmin(mean_distances)
            centroid_index = new_cluster_indices[min_index]

            centroid_indices[assigned_cluster] = centroid_index
            element_indices_by_cluster[assigned_cluster] = new_cluster_indices
        row = torch.cat(
            [
                cluster.tile(element_indices.size(0))
                for cluster, element_indices in zip(
                    torch.arange(k, device=device), element_indices_by_cluster
                )
            ]
        )
        col = torch.cat(element_indices_by_cluster)
        row, col = (col, row) if flow == "source_to_target" else (row, col)

        return torch.stack([row, col], dim=0)

    def kmeans(self, x_0, k=None):
        r"""K-means algorithm wrapper.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, node_features]
            Input features on the nodes of the simplicial complex.

        Returns
        -------
        hyperedge_index : torch.Tensor, shape=[n_nodes, 2]
            Indices of the on-zero values in the feature matrix of hypergraph convolutional network.
        """
        if k is None:
            k = self.k_centroids
        if k > x_0.shape[0]:
            raise ValueError(
                f"There are {x_0.shape[0]} input nodes. The k-means clustering requires at least {k} nodes."
            )
        return self.kmeans_graph(x_0, k=k, flow="source_to_target")

    def get_dynamic_topology(self, x_0_features):
        r"""Dynamic topology computation.

        Parameters
        ----------
        x_0_features : torch.Tensor, shape=[n_nodes, node_features]
            Input features on the nodes of the simplicial complex.

        Returns
        -------
        hyperedge_incidence_matrix : torch.Tensor, shape=[n_nodes, n_nodes + k_centroids]
            Incidence matrix mapping edges to nodes.
        """
        device = x_0_features.device
        n_nodes = x_0_features.size(0)
        batch = torch.zeros(x_0_features.size(0), dtype=torch.long, device=device)
        local_edge_index = knn_graph(
            x_0_features,
            k=self.k_neighbours,
            batch=batch,
            loop=False,
            flow="source_to_target",
        )
        local_hyperedges = torch.sparse_coo_tensor(
            local_edge_index,
            torch.ones(local_edge_index.size(1), device=device),
            size=(n_nodes, n_nodes),
        )  # [n_nodes, n_hyperedges = n_nodes]

        global_edge_index = self.kmeans(x_0_features)
        global_hyperedges = torch.sparse_coo_tensor(
            global_edge_index,
            torch.ones(n_nodes, device=device),
            size=(n_nodes, self.k_centroids),
        )  # [n_nodes, n_hyperedges = k_centroids]
        return torch.cat((local_hyperedges, global_hyperedges), dim=1)

    def forward(self, x_0):
        r"""Forward computation.

        Dynamic topology module of the DHST Block is implemented here.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_nodes, node_channels]
            Input features on the nodes of the simplicial complex.

        Returns
        -------
        x_0 : torch.Tensor, shape=[n_nodes, out_channels]
            Output features on the nodes of the simplicial complex.
        """
        # dynamic topology processing:
        x_0_features = self.fc_layer(x_0)
        incidence_1_dynamic_topology = self.get_dynamic_topology(x_0_features)
        incidence_1_dynamic_topology_transpose = incidence_1_dynamic_topology.transpose(
            1, 0
        )
        x_1 = self.conv_dt_level0_0_to_1(
            x_0_features, incidence_1_dynamic_topology_transpose
        )
        x_0_features = self.conv_dt_level0_1_to_0(x_1, incidence_1_dynamic_topology)
        return x_0_features

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.fc_layer.reset_parameters()
        self.conv_dt_level0_0_to_1.reset_parameters()
        self.conv_dt_level0_1_to_0.reset_parameters()
