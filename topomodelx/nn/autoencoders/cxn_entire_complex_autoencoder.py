import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.nn.cxn.cxn_layers import CXN_AMPS, CXN_CMPS, CXN_HCMPS

"""
Here we implement three versions of entire complex
autoencoders defined on regular cell complexes or simplicial complexes.
The implementation is based on message passing schemes given in
Cell Complex Neural Networks (CXNs):
    https://arxiv.org/abs/2103.04046
    Refs:
        [1] Cell Complex Nueral Networks https://arxiv.org/abs/2010.00743
        [2] Simplicial Complex Representation Learning  https://arxiv.org/abs/2103.04046
"""


class CXN_entire_CX_encoder_AMPS(nn.Module):
    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_AMPS, self).__init__()
        self.dropout = dropout

        self.CXN = CXN_AMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Gv2v, Ge2v, Ge2e, Gf2e, Gf2f):
        """
        Parameters
        ----------
            xv : TYPE. torch tensor of shape
                    [num_nodes, num_node_features]
                DESCRIPTION : representing the input
                feature vector on the nodes of the input SC/CX .
            xe : TYPE. torch tensor of shape
                    [num_edges, num_edges_features]
                DESCRIPTION: representing the input
                feature vector on the edges of
                the input SC/CX.
            xf : TYPE. torch tensor of shape
                [num_faces, num_faces_features]
                DESCRIPTION. representing the input
                feature vector on the faces
                of the input SC/CX.
            Gv2v : TYPE. torch tensor of shape
                [num_nodes , num_nodes ]
                representing the a
                cochain operator C^0->C^0.
            Ge2v : TYPE. torch tensor of shape
                [num_nodes , num_edges]
                representing the a cochain
                operator C^1->C^0.
            Ge2e : TYPE. torch tensor of shape
                [num_edges , num_edges ] representing
                the a cochain operator C^1->C^1.
            Gf2e : TYPE. torch tensor of shape
                [num_edges , num_faces] representing
                the a cochain operator C^2->C^1
            Gf2f : TYPE. torch tensor of shape
                [num_faces , num_faces ] representing
                the a cochain operator C^2->C^2.
        Returns
        -------
            z_mesh : torch tensor of shape [1,n_out]
                DESCRIPTION. A vector embedding that
                represents the entire input complex
        """
        xv, xe, xf = self.CXN(xv, xe, xf, Gv2v, Ge2v, Ge2e, Gf2e, Gf2f)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh


class CXN_entire_CX_encoder_CMPS(nn.Module):
    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_CMPS, self).__init__()
        self.dropout = dropout

        self.CXN = CXN_CMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Gv2v, Gv2e, Ge2e, Ge2f, Gf2f):
        """
        Parameters
        ----------
            xv : TYPE. torch tensor of shape [num_nodes, num_node_features]
                DESCRIPTION : representing the input feature
                vector on the nodes of the input SC/CX .
            xe : TYPE. torch tensor of shape [ num_edges, num_edges_features]
                DESCRIPTION: representing the input feature
                vector on the edges of the input SC/CX
            xf : TYPE. torch tensor of shape [ num_faces, num_faces_features]
                DESCRIPTION. representing the input feature
                vector on the faces of the input SC/CX
            Gv2v : TYPE. torch tensor of shape
                [num_nodes , num_nodes ] representing
                the a cochain operator C^0->C^0.
            Gv2e : TYPE.  torch tensor of shape
                [num_nodes, num_edges] representing
                the a cochain operator C^1->C^2.
            Ge2e : TYPE. torch tensor of shape
                [num_edges , num_edges ] representing
                the a cochain operator C^1->C^1.
            Gf2e : TYPE. torch tensor of shape
                [num_edges , num_faces] representing
                the a cochain operator C^1->C^2.
            Gf2f : TYPE. torch tensor of shape
                [num_faces , num_faces ] representing
                the a cochain operator C^2->C^2.
        Returns
        -------
            z_mesh : torch tensor of shape [1,n_out]
                DESCRIPTION. A vector embedding that represents
                the entire input complex
        """

        xv, xe, xf = self.CXN(xv, xe, xf, Gv2v, Gv2e, Ge2e, Ge2f, Gf2f)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh


class CXN_entire_CX_encoder_HCMPS(nn.Module):
    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_HCMPS, self).__init__()

        self.dropout = dropout

        self.CXN = CXN_HCMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Ge2v, Gf2e):
        """
        Parameters
        ----------
            xv : TYPE. torch tensor of shape
                [num_nodes, num_node_features]
                DESCRIPTION : representing the input
                feature vector on the nodes of the input SC/CX .
            xe : TYPE. torch tensor of shape
                [ num_edges, num_edges_features]
                DESCRIPTION: representing the input
                feature vector on the edges of
                the input SC/CX.
            xf : TYPE. torch tensor of shape
                [num_faces, num_faces_features]
                DESCRIPTION. representing the
                input feature vector on the
                faces of the input SC/CX.
            Ge2v : TYPE. torch tensor of shape
                [num_nodes , num_edges] representing
                the a cochain operator C^1->C^0.
            Gf2e : TYPE. torch tensor of shape
                [num_edges , num_faces] representing
                the a cochain operator C^2->C^1.
        Returns
        -------
            z_mesh : torch tensor of shape [1,n_out]
                DESCRIPTION. A vector embedding that
                represents the entire input complex
        """
        xv, xe, xf = self.CXN(xv, xe, xf, Ge2v, Gf2e)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh
