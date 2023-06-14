"""Template Layer with two conv passing steps."""
import torch
from torch import nn
import torch.nn.functional as F
from topomodelx.base.conv import Conv

# First we need to form the right conv layer, based on topomodelx.base.conv.Conv
class AllSetConv(nn.Module):
    """AllSetConv Layer with MLP and Convolution.

    A layer that applies MLP and convolution operations on the input.

    Parameters
    ----------
    in_dim : int
        Dimension of input features.
    hid_dim : int
        Dimension of intermediate hidden features.
    out_dim : int
        Dimension of output features.
    mlp_num_layers : int
        Number of layers in the MLP.
    dropout : float
        Dropout probability.
    mlp_norm : str, optional
        Normalization technique used in the MLP layers. Defaults to 'ln'.
    input_norm : bool, optional
        Whether to apply input normalization. Defaults to False.
    heads : int, optional
        Number of attention heads. Defaults to 1.
    attention : bool, optional
        Whether to use attention-based propagation. Defaults to False.
    """

    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 mlp_num_layers,
                 dropout,
                 #mlp_norm='ln',
                 input_norm=False,

                 heads=None,
                 attention=False
                 ):
        super(AllSetConv, self).__init__()

        self.attention = attention
        self.dropout = dropout

        # if self.attention:
        #     self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        # else:
        
        if mlp_num_layers > 0:
            self.f_enc = MLP(in_dim, hid_dim, hid_dim, mlp_num_layers, dropout, input_norm)
            self.f_dec = MLP(hid_dim, hid_dim, out_dim, mlp_num_layers, dropout, input_norm)
            in_dim = hid_dim
        else:
            self.f_enc = nn.Identity()
            self.f_dec = nn.Identity()
        
        self.propagate = Conv(
            in_channels=in_dim,
            out_channels=hid_dim,
            aggr_norm=True,
            update_func=None,
        )

    def reset_parameters(self):
        """Reset learnable parameters."""
        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is 'Identity'):
                self.f_dec.reset_parameters()

    def forward(self, x, incidence):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        edge_index : torch.Tensor
            Indices of edges.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """

        if self.attention:
            x = self.prop(x, incidence)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(x, incidence)
            x = F.relu(self.f_dec(x))

        return x

class MLP(nn.Module):
    """MLP Module.

    A multi-layer perceptron module with optional normalization.

    Parameters
    ----------
    in_dim : int
        Dimension of the input features.
    hid_dim : int
        Dimension of the hidden features.
    out_dim : int
        Dimension of the output features.
    num_layers : int
        Number of layers in the MLP.
    dropout : float, optional
        Dropout probability. Defaults to 0.5.
    input_norm : bool, optional
        Whether to apply input normalization. Defaults to False.
    """

    def __init__(self, in_dim, hid_dim, out_dim, num_layers,
                 dropout=.5, input_norm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.input_norm = input_norm
        
        if num_layers == 1:
            # Just a linear layer i.e. logistic regression
            if self.input_norm:
                self.normalizations.append(nn.LayerNorm(in_dim))
            else:
                self.normalizations.append(nn.Identity())
            self.lins.append(nn.Linear(in_dim, out_dim))
        else:
            if self.input_norm:
                self.normalizations.append(nn.LayerNorm(in_dim))
            else:
                self.normalizations.append(nn.Identity())
            self.lins.append(nn.Linear(in_dim, hid_dim))
            self.normalizations.append(nn.LayerNorm(hid_dim))
            for _ in range(num_layers - 2):
                self.lins.append(nn.Linear(hid_dim, hid_dim))
                self.normalizations.append(nn.LayerNorm(hid_dim))
            self.lins.append(nn.Linear(hid_dim, out_dim))
        
        self.dropout = dropout

    def reset_parameters(self):
        """Reset learnable parameters."""
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """

        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x

class AllSetLayer(nn.Module):
    """AllSet Layer Module.

    A module for AllSet layer in a bipartite graph.

    Parameters
    ----------
    in_dim : int
        Dimension of the input features.
    hid_dim : int
        Dimension of the hidden features.
    out_dim : int
        Dimension of the output features.
    dropout : float
        Dropout probability.
    input_dropout : float, optional
        Dropout probability for the layer input. Defaults to 0.2.
    mlp_num_layers : int, optional
        Number of layers in the MLP. Defaults to 2.
    mlp_input_norm : bool, optional
        Whether to apply input normalization in the MLP. Defaults to False.
    heads : int or None, optional
        Number of attention heads. If None, attention is disabled. Defaults to None.
    PMA : bool, optional
        Whether to use the PMA (Prototype Matrix Attention) mechanism. Defaults to False.
    """

    def __init__(self, in_dim, hid_dim, out_dim,
                 dropout,
                 input_dropout=0.2,
                 mlp_num_layers=2,
                 mlp_input_norm=False,
                 heads=None,
                 PMA=False):
        super(AllSetLayer, self).__init__()

        self.dropout = dropout
        self.input_dropout = input_dropout

        self.v2e = AllSetConv(in_dim=in_dim,
                              hid_dim=hid_dim,
                              out_dim=out_dim,
                              mlp_num_layers=mlp_num_layers,
                              dropout=dropout,
                              input_norm=mlp_input_norm,

                              heads=heads,
                              attention=PMA)

        self.e2v = AllSetConv(in_dim=hid_dim,
                              hid_dim=hid_dim,
                              out_dim=out_dim,
                              mlp_num_layers=mlp_num_layers,
                              dropout=dropout,
                              input_norm=mlp_input_norm,

                              heads=heads,
                              attention=PMA)

    def forward(self, x, incidence_1):
        """
        Forward computation.

        Parameters
        ----------
        x : torch.Tensor
            Input features.
        edge_index : torch.Tensor
            Edge list (of size (2, |E|)) where edge_index[0] contains nodes and edge_index[1] contains hyperedges.
        reversed_edge_index : torch.Tensor
            Edge list (of size (2, |E|)) where reversed_edge_index[0] contains hyperedges and reversed_edge_index[1] contains nodes.

        Returns
        -------
        x : torch.Tensor
            Output features.
        """

        x = F.dropout(x, p=self.input_dropout, training=self.training)

        x = F.relu(self.v2e(x, incidence_1.transpose(1, 0)))
        x = F.dropout(x, p=self.dropout, training=self.training)

        x = F.relu(self.e2v(x, incidence_1))
        x = F.dropout(x, p=self.dropout, training=self.training)

        return x 



  # def reset_parameters(self):
    #     for layer in self.V2EConvs:
    #         layer.reset_parameters()
    #     for layer in self.E2VConvs:
    #         layer.reset_parameters()
    #     for layer in self.bnV2Es:
    #         layer.reset_parameters()
    #     for layer in self.bnE2Vs:
    #         layer.reset_parameters()
    #     self.classifier.reset_parameters()



# class TemplateLayer(torch.nn.Module):
#     """Template Layer with two conv passing steps.

#     A two-step message passing layer.

#     Parameters
#     ----------
#     in_channels : int
#         Dimension of input features.
#     intermediate_channels : int
#         Dimension of intermediate features.
#     out_channels : int
#         Dimension of output features.
#     """

#     def __init__(
#         self,
#         in_channels,
#         intermediate_channels,
#         out_channels,
#     ):
#         super().__init__()

#         self.conv_level1_1_to_0 = Conv(
#             in_channels=in_channels,
#             out_channels=intermediate_channels,
#             aggr_norm=True,
#             update_func="sigmoid",
#         )
#         self.conv_level2_0_to_1 = Conv(
#             in_channels=intermediate_channels,
#             out_channels=out_channels,
#             aggr_norm=True,
#             update_func="sigmoid",
#         )

#     def reset_parameters(self):
#         r"""Reset learnable parameters."""
#         self.conv_level1_1_to_0.reset_parameters()
#         self.conv_level2_0_to_1.reset_parameters()

#     def forward(self, x_1, incidence_1):
#         r"""Forward computation.

#         Parameters
#         ----------
#         x_1 : torch.Tensor, shape=[n_edges, in_channels]
#             Input features on the edges of the simplicial complex.
#         incidence_1 : torch.sparse
#             shape=[n_nodes, n_edges]
#             Incidence matrix mapping edges to nodes (B_1).

#         Returns
#         -------
#         x_1 : torch.Tensor, shape=[n_edges, out_channels]
#             Output features on the edges of the simplicial complex.
#         """
#         incidence_1_transpose = incidence_1.transpose(1, 0)
#         if x_1.shape[-2] != incidence_1.shape[-1]:
#             raise ValueError(
#                 f"Shape of input face features does not have the correct number of edges {incidence_1.shape[-1]}."
#             )
#         x_0 = self.conv_level1_1_to_0(x_1, incidence_1)
#         x_1 = self.conv_level2_0_to_1(x_0, incidence_1_transpose)
#         return x_1
    