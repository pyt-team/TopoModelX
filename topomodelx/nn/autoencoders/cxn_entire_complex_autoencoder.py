"""Autoencoders on topological domains.

Here we implement three versions of entire complex
autoencoders defined on regular cell complexes or simplicial complexes.

The implementation is based on message passing schemes given in
Cell Complex Neural Networks (CXNs):
https://arxiv.org/abs/2103.04046

References
----------
[1] Cell Complex Neural Networks https://arxiv.org/abs/2010.00743
[2] Simplicial Complex Representation Learning https://arxiv.org/abs/2103.04046
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

from topomodelx.nn.cxn.cxn_layers import CXN_AMPS, CXN_CMPS, CXN_HCMPS


class CXN_entire_CX_encoder_AMPS(nn.Module):
    """Autoencoder on entire complex.

    This implements the message passing AMPS.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    n_hid : int
    n_out : int
        Embedding dimension.
    dropout : float, optional
        Dropout rate. The default is 0.05.
    alpha : float, optional
        Negative slope of the LeakyReLU activation function. The default is 0.1.
    """

    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_AMPS, self).__init__()
        self.dropout = dropout

        self.CXN = CXN_AMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Gv2v, Ge2v, Ge2e, Gf2e, Gf2f):
        """Forward pass.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Input feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[n_edges, n_edges_features]
            Input feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Input feature vector on the faces of the input SC/CX.
        Gv2v : torch.tensor, shape=[n_nodes, n_nodes]
            Cochain operator C^0->C^0.
        Ge2v : torch.tensor, shape=[n_nodes, n_edges]
            Cochain operator C^1->C^0.
        Ge2e : torch.tensor, shape=[n_edges, n_edges]
            Cochain operator C^1->C^1.
        Gf2e : torch.tensor, shape=[n_edges, n_faces]
            Cochain operator C^2->C^1
        Gf2f : torch.tensor, shape=[n_faces, n_faces]
            Cochain operator C^2->C^2.

        Returns
        -------
        z_mesh : torch.tensor, shape=[1, n_out]
            Vector embedding that represents the entire input complex.
        """
        xv, xe, xf = self.CXN(xv, xe, xf, Gv2v, Ge2v, Ge2e, Gf2e, Gf2f)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh


class CXN_entire_CX_encoder_CMPS(nn.Module):
    """Autoencoder on entire complex.

    This implements the message passing CMPS.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    n_hid : int
    n_out : int
        Embedding dimension.
    dropout : float, optional
        Dropout rate. The default is 0.05.
    alpha : float, optional
        Negative slope of the LeakyReLU activation function. The default is 0.1.
    """

    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_CMPS, self).__init__()
        self.dropout = dropout

        self.CXN = CXN_CMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Gv2v, Gv2e, Ge2e, Ge2f, Gf2f):
        """Forward pass.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[n_edges, n_edges_features]
            Feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Feature vector on the faces of the input SC/CX.
        Gv2v : torch.tensor, shape=[n_nodes , n_nodes ]
            Cochain operator C^0->C^0.
        Gv2e : toch.tensor, shape=[n_nodes, n_edges]
            Cochain operator C^1->C^2.
        Ge2e : torch.tensor, shape=[n_edges , n_edges ]
            Cochain operator C^1->C^1.
        Gf2e : torch.tensor, shape=[n_edges , n_faces]
            Cochain operator C^1->C^2.
        Gf2f : torch.tensor, shape=[n_faces , n_faces ]
            Cochain operator C^2->C^2.

        Returns
        -------
        z_mesh : torch.tensor, shape=[1,n_out]
            Vector embedding that represents the entire input complex
        """
        xv, xe, xf = self.CXN(xv, xe, xf, Gv2v, Gv2e, Ge2e, Ge2f, Gf2f)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh


class CXN_entire_CX_encoder_HCMPS(nn.Module):
    """Autoencoder on entire complex.

    This implements the message passing HCMPS.

    Parameters
    ----------
    in_ch : int
        Number of input channels.
    n_hid : int
    n_out : int
        Embedding dimension.
    dropout : float, optional
        Dropout rate. The default is 0.05.
    alpha : float, optional
        Negative slope of the LeakyReLU activation function. The default is 0.1.
    """

    def __init__(self, in_ch, n_hid, n_out, dropout=0.05, alpha=0.1):
        super(CXN_entire_CX_encoder_HCMPS, self).__init__()

        self.dropout = dropout

        self.CXN = CXN_HCMPS(in_ch, in_ch, in_ch, n_hid, n_hid, n_hid)

        self.linear = nn.Linear(n_hid, n_out)

        self.act = nn.LeakyReLU(alpha)

        self.act1 = nn.Sigmoid()

        self.act2 = nn.ReLU()

    def forward(self, xv, xe, xf, Ge2v, Gf2e):
        """Forward pass.

        Parameters
        ----------
        xv : torch.tensor, shape=[n_nodes, n_node_features]
            Feature vector on the nodes of the input SC/CX.
        xe : torch.tensor, shape=[ n_edges, n_edges_features]
            Feature vector on the edges of the input SC/CX.
        xf : torch.tensor, shape=[n_faces, n_faces_features]
            Feature vector on the faces of the input SC/CX.
        Ge2v : torch.tensor, shape=[n_nodes , n_edges]
            Cochain operator C^1->C^0.
        Gf2e : torch.tensor, shape=[n_edges , n_faces]
            Cochain operator C^2->C^1.

        Returns
        -------
        z_mesh : torch.tensor, shape=[1,n_out]
            Vector embedding that represents the entire input complex.
        """
        xv, xe, xf = self.CXN(xv, xe, xf, Ge2v, Gf2e)

        z_mesh = torch.cat((xv, xe, xf), dim=1)

        z_mesh = torch.mean(z_mesh, 1)

        z_mesh = F.dropout(z_mesh, self.dropout, self.training)

        z_mesh = self.act2(z_mesh)

        z_mesh = self.linear(z_mesh)

        return z_mesh
