"""Test the BSC Layer."""
import networkx as nx
import numpy as np
import pytest
import scipy.sparse as sp
import torch
import torch_geometric
import torch_geometric.transforms as T
from numpy.linalg import inv, pinv

from topomodelx.nn.simplicial.bScNet_layer import BlockNet

# from topomodelx.nn.simplicial.bScNet_layer import testData


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
            root="tmp/Cora", name="Cora", transform=T.NormalizeFeatures()
        )
        data2 = dataset[0]

        testD, num_features, num_classes, boundary_matrics = testData(
            data2, dataset.name, data2.x.size(1), dataset.num_classes
        )
        # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        bscn = BlockNet(testD, num_features, num_classes, boundary_matrics)

        emb = bscn.g_encode(testD)
        # # hsn = BSCLayer(channels)
        # # output = hsn.forward(x_0, incidence_1, adjacency_0)
        # # print(emb)
        assert emb.shape[0] >= 1

    def test_reset_parameters(self):
        """Test the reset of the parameters."""
        # channels = 5

        # hsn = BSCLayer(channels)
        # hsn.reset_parameters()
        # dataset = torch_geometric.datasets.Planetoid(
        #     root="tmp/Cora", name="Cora", transform=T.NormalizeFeatures()
        # )
        # data = dataset[0]

        # testD, num_features, num_classes, boundary_matrics = testData(
        #     data, dataset.name, data.x.size(1), dataset.num_classes
        # )
        # # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        # bscn = BlockNet(testD, num_features, num_classes, boundary_matrics)
        # bscn.reset_parameters()

        # for module in bscn.modules():
        #     if isinstance(module, torch.nn.Conv2d):
        #         torch.testing.assert_allclose(
        #             module.weight, torch.zeros_like(module.weight)
        #         )
        #         torch.testing.assert_allclose(
        #             module.bias, torch.zeros_like(module.bias)
        #         )

        # for module in hsn.modules():
        #     if isinstance(module, torch.nn.Conv2d):
        #         torch.testing.assert_allclose(
        #             module.weight, torch.zeros_like(module.weight)
        #         )
        #         torch.testing.assert_allclose(
        #             module.bias, torch.zeros_like(module.bias)
        #         )


@pytest.mark.skip(reason="Utility function to get data")
def testData(cdata, name, num_features, num_classes):
    val_prop = 0.05
    test_prop = 0.1
    (
        train_edges,
        train_edges_false,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    ) = get_edges_split(cdata, val_prop=val_prop, test_prop=test_prop)
    total_edges = np.concatenate(
        (
            train_edges,
            train_edges_false,
            val_edges,
            val_edges_false,
            test_edges,
            test_edges_false,
        )
    )
    cdata.train_pos, cdata.train_neg = len(train_edges), len(train_edges_false)
    cdata.val_pos, cdata.val_neg = len(val_edges), len(val_edges_false)
    cdata.test_pos, cdata.test_neg = len(test_edges), len(test_edges_false)
    cdata.total_edges = total_edges
    cdata.total_edges_y = torch.cat(
        (
            torch.ones(len(train_edges)),
            torch.zeros(len(train_edges_false)),
            torch.ones(len(val_edges)),
            torch.zeros(len(val_edges_false)),
            torch.ones(len(test_edges)),
            torch.zeros(len(test_edges_false)),
        )
    ).long()

    # delete val_pos and train_pos
    edge_list = np.array(cdata.edge_index).T.tolist()
    for edges in val_edges:
        edges = edges.tolist()
        if edges in edge_list:
            # if not in edge_list, mean it is a self loop
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    for edges in train_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    cdata.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)

    # edge index sampling
    random_edge_num = 500
    indices = np.random.choice(
        (cdata.edge_index).size(1), (random_edge_num,), replace=False
    )
    indices = np.sort(indices)
    sample_data_edge_index = cdata.edge_index[:, indices]

    boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(
        cdata, sample_data_edge_index
    )
    boundary_matrices_option = False

    # if boundary_matrices_option:
    #     # convert hodge matrix to tensor format
    boundary_matrix0 = torch.tensor(boundary_matrix0_, dtype=torch.float32)
    boundary_matrix1 = torch.tensor(boundary_matrix1_, dtype=torch.float32)
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cdata.total_edges_y
    #     model, data = BlockNet(data, num_features, num_classes,
    #                            boundary_matrics=[boundary_matrix0.to(device), boundary_matrix1.to(device)]).to(device), data.to(device)

    # else:
    #     L0u, L1f = compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
    #     # convert hodge matrix to tensor format
    #     L0u = torch.tensor(L0u, dtype=torch.float32)
    #     L1f = torch.tensor(L1f, dtype=torch.float32)
    #     device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    #     data.total_edges_y.to(device)
    #     model, data = BlockNet(data, num_features, num_classes,
    #                            boundary_matrics=[L0u.to(device), L1f.to(device)]).to(device), data.to(device)
    boundary_matrics = [boundary_matrix0, boundary_matrix1]

    return cdata, num_features, num_classes, boundary_matrics


def get_edges_split(cdata2, val_prop=0.2, test_prop=0.2):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(cdata2.y))])
    _edge_index_ = np.array((cdata2.edge_index))
    edge_index_ = [
        (_edge_index_[0, i], _edge_index_[1, i])
        for i in range(np.shape(_edge_index_)[1])
    ]
    g.add_edges_from(edge_index_)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj, val_prop=val_prop, test_prop=test_prop)


def get_adj_split(adj, val_prop=0.05, test_prop=0.1):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1.0 - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = (
        pos_edges[:n_val],
        pos_edges[n_val : n_test + n_val],
        pos_edges[n_test + n_val :],
    )
    val_edges_false, test_edges_false = (
        neg_edges[:n_val],
        neg_edges[n_val : n_test + n_val],
    )
    train_edges_false = np.concatenate([neg_edges, val_edges, test_edges], axis=0)
    return (
        train_edges,
        train_edges_false,
        val_edges,
        val_edges_false,
        test_edges,
        test_edges_false,
    )


def compute_hodge_matrix(cdata2, sample_data_edge_index):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(cdata2.y))])
    edge_index_ = np.array((sample_data_edge_index))
    edge_index = [
        (edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])
    ]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(
        g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx
    )

    return B1, B2


def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i + 1, len(edges)):
            e1 = edges[i]
            e2 = edges[j]
            if e1[0] == e2[0]:
                shared = e1[0]
                e3 = (e1[1], e2[1])
            elif e1[1] == e2[0]:
                shared = e1[1]
                e3 = (e1[0], e2[1])
            elif e1[0] == e2[1]:
                shared = e1[0]
                e3 = (e1[1], e2[0])
            elif e1[1] == e2[1]:
                shared = e1[1]
                e3 = (e1[0], e2[0])
            else:  # edges don't connect
                continue

            if e3[0] in G[e3[1]]:  # if 3rd edge is in graph
                faces.append(tuple(sorted((shared, *e3))))
    return list(sorted(set(faces)))


def incidence_matrices(G, V, E, faces, edge_to_idx):
    """
    Returns incidence matrices B1 and B2

    :param G: NetworkX DiGraph
    :param V: list of nodes
    :param E: list of edges
    :param faces: list of faces in G

    Returns B1 (|V| x |E|) and B2 (|E| x |faces|)
    B1[i][j]: -1 if node is is tail of edge j, 1 if node is head of edge j, else 0 (tail -> head) (smaller -> larger)
    B2[i][j]: 1 if edge i appears sorted in face j, -1 if edge i appears reversed in face j, else 0; given faces with sorted node order
    """
    B1 = np.array(
        nx.incidence_matrix(G, nodelist=V, edgelist=E, oriented=True).todense()
    )
    B2 = np.zeros([len(E), len(faces)])

    for f_idx, face in enumerate(faces):  # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


def compute_hodge_basis_matrices(cdata2):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(cdata2.y))])
    edge_index_ = np.array((cdata2.edge_index))
    edge_index = [
        (edge_index_[0, i], edge_index_[1, i]) for i in range(np.shape(edge_index_)[1])
    ]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(
        g, sorted(g.nodes), sorted(g.edges), get_faces(g), edge_to_idx
    )

    return B1, B2


def compute_D2(B):
    """
    Computes D2 = max(diag(dot(|B|, 1)), I)
    """
    B_rowsum = np.abs(B).sum(axis=1)

    D2 = np.diag(np.maximum(B_rowsum, 1))
    return D2


def compute_D1(B1, D2):
    """
    Computes D1 = 2 * max(diag(|B1|) .* D2
    """
    rowsum = (np.abs(B1) @ D2).sum(axis=1)
    D1 = 2 * np.diag(rowsum)

    return D1


def compute_hodge_laplacian(B1, B2):
    """
    Computes normalized A0 and A1 matrices (up and down),
        and returns all matrices needed for Bunch model shift operators
    """
    # print(B1.shape, B2.shape)

    # D matrices
    D2_2 = compute_D2(B2)
    D1 = compute_D1(B1, D2_2)
    D3 = np.identity(B2.shape[1]) / 3  # (|F| x |F|)

    # L matrices
    D1_pinv = pinv(D1)
    D2_2_inv = inv(D2_2)

    L1u = D2_2 @ B1.T @ D1_pinv @ B1
    L1d = B2 @ D3 @ B2.T @ D2_2_inv
    L1f = L1u + L1d

    return L1f


# dataset = torch_geometric.datasets.Planetoid(
#     root="tmp/Cora", name="Cora", transform=T.NormalizeFeatures()
# )
# data = dataset[0]

# print(data)

# testD, num_features, num_classes, boundary_matrics = testData(
#     data, dataset.name, data.x.size(1), dataset.num_classes
# )
# print(testD)
# bscn = BlockNet(testD, num_features, num_classes, boundary_matrics)

# emb = bscn.g_encode(testD)
