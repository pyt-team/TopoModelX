import math
from numpy.linalg import inv, pinv
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_geometric.transforms as T
import numpy as np
from torch.nn import Softmax
from torch_geometric.nn import GCNConv
import networkx as nx
from scipy.spatial import Delaunay
import scipy.sparse as sp



class BlockNet(torch.nn.Module):
    def __init__(self, data, num_features, num_classes, boundary_matrics, dimension=8):
        super(BlockNet, self).__init__()
        self.conv1 = GCNConv(num_features, 64, cached=True)
        self.conv2 = GCNConv(64, 32, cached=True)
        self.boundary_matrics = boundary_matrics
        self.leakyrelu = torch.nn.LeakyReLU(0.2, True)
        self.linear = torch.nn.Linear(32, 1, bias=True)
        self.linear_1 = torch.nn.Linear(dimension + 32, 32, bias=True)
        self.softmax = Softmax(dim=1)
        boundary_matrix_size = self.boundary_matrics[0].size(0)
        self.weights_sim = nn.Parameter(torch.FloatTensor(
            int(boundary_matrix_size*2), dimension))
        self.embeddings_sim = nn.Parameter(torch.FloatTensor(
            data.x.size(1), int(boundary_matrix_size*2)))
        self.weights_off_diagonal = nn.Parameter(torch.FloatTensor(
            int(boundary_matrix_size), int(boundary_matrix_size)))
        self.weights_L_0 = nn.Parameter(
            torch.FloatTensor(int(boundary_matrix_size), 32))
        self.weights_L_1 = nn.Parameter(
            torch.FloatTensor(int(boundary_matrix_size), 32))
        # reset parameters
        nn.init.kaiming_uniform_(
            self.weights_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(
            self.embeddings_sim, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(
            self.weights_off_diagonal, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(
            self.weights_L_0, mode='fan_out', a=math.sqrt(5))
        nn.init.kaiming_uniform_(
            self.weights_L_1, mode='fan_out', a=math.sqrt(5))

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv1.reset_parameters()
        self.conv2.reset_parameters()
        
        
    def g_encode(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = F.relu(self.conv2(x, edge_index))
        return x

    def s_encode(self, data, g_emb, type="train"):
        if type == 'train':
            edges_pos = data.total_edges[:data.train_pos]
            index = np.random.randint(0, data.train_neg, data.train_pos)
            edges_neg = data.total_edges[data.train_pos:
                                         data.train_pos + data.train_neg][index]
            total_edges = np.concatenate((edges_pos, edges_neg))
            edges_y = torch.cat(
                (data.total_edges_y[:data.train_pos], data.total_edges_y[data.train_pos:data.train_pos + data.train_neg][index]))

        elif type == 'val':
            total_edges = data.total_edges[data.train_pos +
                                           data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]
            edges_y = data.total_edges_y[data.train_pos +
                                         data.train_neg:data.train_pos+data.train_neg+data.val_pos+data.val_neg]

        elif type == 'test':
            total_edges = data.total_edges[
                data.train_pos + data.train_neg + data.val_pos + data.val_neg:]
            edges_y = data.total_edges_y[
                data.train_pos + data.train_neg + data.val_pos + data.val_neg:]

        L0, L1 = self.boundary_matrics
        zeros_off_diagonal = False

        if zeros_off_diagonal:
            L0_r = torch.matrix_power(L0, 1)
            L1_r = torch.matrix_power(L1, 1)
            diag_zeros = torch.zeros(L0.size()).to(L0.device)
            upper_block = torch.cat([L0_r, diag_zeros], dim=1)
            lower_block = torch.cat([diag_zeros, L1_r], dim=1)
            sim_block = torch.cat([upper_block, lower_block], dim=0)
        else:
            L0_r = torch.matrix_power(L0, 2)
            L1_r = torch.matrix_power(L1, 2)
            relation_embedded = torch.einsum('xd, dy -> xy', torch.matmul(
                L0_r, self.weights_L_0), torch.matmul(L1_r, self.weights_L_1).transpose(0, 1))
            relation_embedded_ = torch.matmul(
                self.weights_off_diagonal, relation_embedded)
            upper_block = torch.cat([L0_r, relation_embedded_], dim=1)
            lower_block = torch.cat(
                [torch.transpose(relation_embedded_, 0, 1), L1_r], dim=1)
            sim_block = torch.cat([upper_block, lower_block], dim=0)
            sim_block = F.softmax(F.relu(sim_block), dim=1)

        x = data.x
        embeddings_sim = torch.matmul(x, self.embeddings_sim)
        s_emb_sim_ = torch.matmul(embeddings_sim, sim_block)
        s_emb_sim = torch.matmul(s_emb_sim_, self.weights_sim)
        s_emb_sim = s_emb_sim.renorm_(2, 0, 1)

        s_emb_sim_in = s_emb_sim[total_edges[:, 0]]
        s_emb_sim_out = s_emb_sim[total_edges[:, 1]]

        d_sim = (s_emb_sim_in - s_emb_sim_out).pow(2)

        # linear to gather edge features
        # embedding from GCN
        g_emb = g_emb.renorm_(2, 0, 1)
        alpha = 1.0
        beta = 0.1

        g_emb_in = g_emb[total_edges[:, 0]]
        g_emb_out = g_emb[total_edges[:, 1]]
        g_sqdist = (g_emb_in - g_emb_out).pow(2)
        sqdist = self.leakyrelu(self.linear_1(
            torch.cat((alpha * g_sqdist, beta * d_sim), dim=1)))
        sqdist = torch.abs(self.linear(sqdist)).reshape(-1)
        sqdist = torch.clamp(sqdist, min=0, max=40)
        prob = 1. / (torch.exp((sqdist - 2.0) / 1.0) + 1.0)
        return prob, edges_y.float()


def num(strings):
    try:
        return int(strings)
    except ValueError:
        return float(strings)


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


def compute_bunch_matrices(B1, B2):
    """
    Computes normalized A0 and A1 matrices (up and down),
        and returns all matrices needed for Bunch model shift operators
    """
    # D matrices
    D2_2 = compute_D2(B2)
    D2_1 = compute_D2(B1)
    D3_n = np.identity(B1.shape[1])  # (|E| x |E|)
    D1 = compute_D1(B1, D2_2)
    D3 = np.identity(B2.shape[1]) / 3  # (|F| x |F|)

    # L matrices
    D1_pinv = pinv(D1)
    D2_2_inv = inv(D2_2)

    L0u = B1.T @ B1  # B1 @ D3_n @ B1.T @ inv(D2_1)
    L1u = D2_2 @ B1.T @ D1_pinv @ B1
    L1d = B2 @ D3 @ B2.T @ D2_2_inv
    L1f = L1u + L1d

    return L0u, L1f


def testData(data, name, num_features, num_classes):
    val_prop = 0.05
    test_prop = 0.1
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(
        data, val_prop=val_prop, test_prop=test_prop)
    total_edges = np.concatenate(
        (train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false))
    data.train_pos, data.train_neg = len(train_edges), len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat((torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)), torch.ones(len(
        val_edges)), torch.zeros(len(val_edges_false)), torch.ones(len(test_edges)), torch.zeros(len(test_edges_false)))).long()

    # delete val_pos and train_pos
    edge_list = np.array(data.edge_index).T.tolist()
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
    data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)

    # edge index sampling
    random_edge_num = 500
    indices = np.random.choice((data.edge_index).size(
        1), (random_edge_num,), replace=False)
    indices = np.sort(indices)
    sample_data_edge_index = data.edge_index[:, indices]

    boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(
        data, sample_data_edge_index)
    boundary_matrices_option = False

    # if boundary_matrices_option:
    #     # convert hodge matrix to tensor format
    boundary_matrix0 = torch.tensor(boundary_matrix0_, dtype=torch.float32)
    boundary_matrix1 = torch.tensor(boundary_matrix1_, dtype=torch.float32)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    data.total_edges_y.to(device)
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
    boundary_matrics = [boundary_matrix0.to(device), boundary_matrix1.to(device)].to(device)
    return data, num_features, num_classes, boundary_matrics

def call(data, name, num_features, num_classes):
    if name in ['PPI']:
        val_prop = 0.2
        test_prop = 0.2
    else:
        val_prop = 0.05
        test_prop = 0.1
    train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false = get_edges_split(
        data, val_prop=val_prop, test_prop=test_prop)
    total_edges = np.concatenate(
        (train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false))
    data.train_pos, data.train_neg = len(train_edges), len(train_edges_false)
    data.val_pos, data.val_neg = len(val_edges), len(val_edges_false)
    data.test_pos, data.test_neg = len(test_edges), len(test_edges_false)
    data.total_edges = total_edges
    data.total_edges_y = torch.cat((torch.ones(len(train_edges)), torch.zeros(len(train_edges_false)), torch.ones(len(
        val_edges)), torch.zeros(len(val_edges_false)), torch.ones(len(test_edges)), torch.zeros(len(test_edges_false)))).long()

    # delete val_pos and test_pos
    edge_list = np.array(data.edge_index).T.tolist()
    for edges in val_edges:
        edges = edges.tolist()
        if edges in edge_list:
            # if not in edge_list, mean it is a self loop
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    for edges in test_edges:
        edges = edges.tolist()
        if edges in edge_list:
            edge_list.remove(edges)
            edge_list.remove([edges[1], edges[0]])
    data.edge_index = torch.Tensor(edge_list).long().transpose(0, 1)

    # edge index sampling
    random_edge_num = 2500
    indices = np.random.choice((data.edge_index).size(
        1), (random_edge_num,), replace=False)
    indices = np.sort(indices)
    sample_data_edge_index = data.edge_index[:, indices]

    boundary_matrix0_, boundary_matrix1_ = compute_hodge_matrix(
        data, sample_data_edge_index)
    boundary_matrices_option = False

    if boundary_matrices_option:
        # convert hodge matrix to tensor format
        boundary_matrix0 = torch.tensor(boundary_matrix0_, dtype=torch.float32)
        boundary_matrix1 = torch.tensor(boundary_matrix1_, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.total_edges_y.to(device)
        model, data = BlockNet(data, num_features, num_classes,
                               boundary_matrics=[boundary_matrix0.to(device), boundary_matrix1.to(device)]).to(device), data.to(device)

    else:
        L0u, L1f = compute_bunch_matrices(boundary_matrix0_, boundary_matrix1_)
        # convert hodge matrix to tensor format
        L0u = torch.tensor(L0u, dtype=torch.float32)
        L1f = torch.tensor(L1f, dtype=torch.float32)
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        data.total_edges_y.to(device)
        model, data = BlockNet(data, num_features, num_classes,
                               boundary_matrics=[L0u.to(device), L1f.to(device)]).to(device), data.to(device)

    return model, data


def get_edges_split(data, val_prop=0.2, test_prop=0.2):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    _edge_index_ = np.array((data.edge_index))
    edge_index_ = [(_edge_index_[0, i], _edge_index_[1, i]) for i in
                   range(np.shape(_edge_index_)[1])]
    g.add_edges_from(edge_index_)
    adj = nx.adjacency_matrix(g)

    return get_adj_split(adj, val_prop=val_prop, test_prop=test_prop)


def get_adj_split(adj, val_prop=0.05, test_prop=0.1):
    x, y = sp.triu(adj).nonzero()
    pos_edges = np.array(list(zip(x, y)))
    np.random.shuffle(pos_edges)
    # get tn edges
    x, y = sp.triu(sp.csr_matrix(1. - adj.toarray())).nonzero()
    neg_edges = np.array(list(zip(x, y)))
    np.random.shuffle(neg_edges)

    m_pos = len(pos_edges)
    n_val = int(m_pos * val_prop)
    n_test = int(m_pos * test_prop)
    val_edges, test_edges, train_edges = pos_edges[:n_val], pos_edges[n_val:n_test +
                                                                      n_val], pos_edges[n_test + n_val:]
    val_edges_false, test_edges_false = neg_edges[:
                                                  n_val], neg_edges[n_val:n_test + n_val]
    train_edges_false = np.concatenate(
        [neg_edges, val_edges, test_edges], axis=0)
    return train_edges, train_edges_false, val_edges, val_edges_false, test_edges, test_edges_false


def compute_hodge_matrix(data, sample_data_edge_index):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    edge_index_ = np.array((sample_data_edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                  range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(
        g.edges), get_faces(g), edge_to_idx)

    return B1, B2


def get_faces(G):
    """
    Returns a list of the faces in an undirected graph
    """
    edges = list(G.edges)
    faces = []
    for i in range(len(edges)):
        for j in range(i+1, len(edges)):
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
    B1 = np.array(nx.incidence_matrix(
        G, nodelist=V, edgelist=E, oriented=True).todense())
    B2 = np.zeros([len(E), len(faces)])

    for f_idx, face in enumerate(faces):  # face is sorted
        edges = [face[:-1], face[1:], [face[0], face[2]]]
        e_idxs = [edge_to_idx[tuple(e)] for e in edges]

        B2[e_idxs[:-1], f_idx] = 1
        B2[e_idxs[-1], f_idx] = -1
    return B1, B2


def compute_hodge_basis_matrices(data):
    g = nx.Graph()
    g.add_nodes_from([i for i in range(len(data.y))])
    edge_index_ = np.array((data.edge_index))
    edge_index = [(edge_index_[0, i], edge_index_[1, i]) for i in
                  range(np.shape(edge_index_)[1])]
    g.add_edges_from(edge_index)

    edge_to_idx = {edge: i for i, edge in enumerate(g.edges)}

    B1, B2 = incidence_matrices(g, sorted(g.nodes), sorted(
        g.edges), get_faces(g), edge_to_idx)

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
