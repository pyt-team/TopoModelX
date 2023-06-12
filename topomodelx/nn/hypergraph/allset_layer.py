"""Template Layer with two conv passing steps."""
import torch
from torch import nn
import torch.nn.functional as F
from topomodelx.base.conv import Conv


class TemplateLayer(torch.nn.Module):
    """Template Layer with two conv passing steps.

    A two-step message passing layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    intermediate_channels : int
        Dimension of intermediate features.
    out_channels : int
        Dimension of output features.
    """

    def __init__(
        self,
        in_channels,
        intermediate_channels,
        out_channels,
    ):
        super().__init__()

        self.conv_level1_1_to_0 = Conv(
            in_channels=in_channels,
            out_channels=intermediate_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )
        self.conv_level2_0_to_1 = Conv(
            in_channels=intermediate_channels,
            out_channels=out_channels,
            aggr_norm=True,
            update_func="sigmoid",
        )

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        self.conv_level1_1_to_0.reset_parameters()
        self.conv_level2_0_to_1.reset_parameters()

    def forward(self, x_1, incidence_1):
        r"""Forward computation.

        Parameters
        ----------
        x_1 : torch.Tensor, shape=[n_edges, in_channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_1 : torch.Tensor, shape=[n_edges, out_channels]
            Output features on the edges of the simplicial complex.
        """
        incidence_1_transpose = incidence_1.transpose(1, 0)
        if x_1.shape[-2] != incidence_1.shape[-1]:
            raise ValueError(
                f"Shape of input face features does not have the correct number of edges {incidence_1.shape[-1]}."
            )
        x_0 = self.conv_level1_1_to_0(x_1, incidence_1)
        x_1 = self.conv_level2_0_to_1(x_0, incidence_1_transpose)
        return x_1
    


class SetGNN(nn.Module):
    def __init__(self, in_dim, hid_dim, out_dim,
                 allset_num_layers,
                 input_dropour,
                 dropout,
                 


                 mlp_num_layers=2, 
                 mlp_input_norm=False,
                 
                 norm_layer_type='ln',
                 heads=None,
                 PMA=False,):
        super(SetGNN, self).__init__()
        """
        args should contain the following:
        V_in_dim, V_enc_hid_dim, V_dec_hid_dim, V_out_dim, V_enc_num_layers, V_dec_num_layers
        E_in_dim, E_enc_hid_dim, E_dec_hid_dim, E_out_dim, E_enc_num_layers, E_dec_num_layers
        All_num_layers,dropout
        !!! V_in_dim should be the dimension of node features
        !!! E_out_dim should be the number of classes (for classification)
        """    
        
        
#         Now define V2EConvs[i], V2EConvs[i] for ith layers
#         Currently we assume there's no hyperedge features, which means V_out_dim = E_in_dim
#         If there's hyperedge features, concat with Vpart decoder output features [V_feat||E_feat]
        self.v2e, self.e2v = nn.ModuleList(), nn.ModuleList()
        self.dropout = dropout
    

        # There is not possibility to learn feature importance for now
        #  if self.LearnMask:
        #     self.Importance = nn.Parameter(torch.ones(norm.size()))

        
        for layer_idx in range(self.allset_num_layers):

            if layer_idx==0:
                self.v2e.append(SetConvLayer(in_dim=in_dim,
                                            hid_dim=hid_dim,
                                            out_dim=out_dim,
                                            num_layers=mlp_num_layers,
                                            dropout=dropout,
                                            Normalization=norm_layer_type,
                                            InputNorm=mlp_input_norm,
                                            heads=heads,
                                            attention=PMA))
            else:
                self.v2e.append(SetConvLayer(in_dim=in_dim,
                                            hid_dim=hid_dim,
                                            out_dim=out_dim,
                                            num_layers=mlp_num_layers,
                                            dropout=dropout,
                                            Normalization=norm_layer_type,
                                            InputNorm=mlp_input_norm,
                                            heads=heads,
                                            attention=PMA))
            
            
            self.e2v.append(SetConvLayer(in_dim=in_dim,
                                            hid_dim=hid_dim,
                                            out_dim=out_dim,
                                            num_layers=mlp_num_layers,
                                            dropout=dropout,
                                            Normalization=norm_layer_type,
                                            InputNorm=mlp_input_norm,
                                            heads=heads,
                                            attention=PMA))
        
            

    def reset_parameters(self):
        for layer in self.V2EConvs:
            layer.reset_parameters()
        for layer in self.E2VConvs:
            layer.reset_parameters()
        for layer in self.bnV2Es:
            layer.reset_parameters()
        for layer in self.bnE2Vs:
            layer.reset_parameters()
        self.classifier.reset_parameters()
        

    def forward(self, x, edge_index):
        """
        The data should contain the follows
        data.x: node features
        data.edge_index: edge list (of size (2,|E|)) where data.edge_index[0] contains nodes and data.edge_index[1] contains hyperedges
        !!! Note that self loop should be assigned to a new (hyper)edge id!!!
        !!! Also note that the (hyper)edge id should start at 0 (akin to node id)
        data.norm: The weight for edges in bipartite graphs, correspond to data.edge_index
        !!! Note that we output final node representation. Loss should be defined outside.
        """
#             The data should contain the follows
#             data.x: node features
#             data.V2Eedge_index:  edge list (of size (2,|E|)) where
#             data.V2Eedge_index[0] contains nodes and data.V2Eedge_index[1] contains hyperedges
        
        cidx = edge_index[1].min()
        edge_index[1] -= cidx  # make sure we do not waste memory
        reversed_edge_index = torch.stack(
            [edge_index[1], edge_index[0]], dim=0)
        
        # Input dropout (following original implementation)
        x = F.dropout(x, p=0.2, training=self.training) 
        for i, _ in enumerate(self.v2e):
            # Updated hyperedges
            x = F.relu(self.v2e[i](x, edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)
            # Updated nodes
            x = F.relu(self.e2v[i](x, reversed_edge_index))
            x = F.dropout(x, p=self.dropout, training=self.training)


        return x



class SetConvLayer():
    def __init__(self,
                 in_dim,
                 hid_dim,
                 out_dim,
                 mlp_num_layers,
                 dropout,
                 mlp_norm='ln',
                 InputNorm=False,
                 heads=1,
                 attention=False
                 ):
        super(SetConvLayer, self).__init__()

        self.attention = attention
        self.dropout = dropout

        #if self.attention:
            #self.prop = PMA(in_dim, hid_dim, out_dim, num_layers, heads=heads)
        #else:
        self.propagate = Conv(
            in_channels=in_dim,
            out_channels=hid_dim,
            aggr_norm=True,
            update_func=None,
        )
        if mlp_num_layers > 0:
            self.f_enc = MLP(in_dim, hid_dim, hid_dim, mlp_num_layers, dropout, mlp_norm, InputNorm)
            self.f_dec = MLP(hid_dim, hid_dim, out_dim, mlp_num_layers, dropout, mlp_norm, InputNorm)
        else:
            self.f_enc = nn.Identity()
            self.f_dec = nn.Identity()

    def reset_parameters(self):

        if self.attention:
            self.prop.reset_parameters()
        else:
            if not (self.f_enc.__class__.__name__ is 'Identity'):
                self.f_enc.reset_parameters()
            if not (self.f_dec.__class__.__name__ is 'Identity'):
                self.f_dec.reset_parameters()

    def forward(self, x, edge_index):
        """
        input -> MLP -> Conv Layer -> MLP -> output
        """
        
        if self.attention:
            x = self.prop(x, edge_index)
        else:
            x = F.relu(self.f_enc(x))
            x = F.dropout(x, p=self.dropout, training=self.training)
            x = self.propagate(x_source=x, neighborhood=edge_index) #self.propagate(edge_index, x=x, norm=norm, aggr=aggr)
            x = F.relu(self.f_dec(x))
            
        return x

#     def message(self, x_j, norm):
#         return norm.view(-1, 1) * x_j

#     def aggregate(self, inputs, index,
#                   dim_size=None, aggr=None):
#         r"""Aggregates messages from neighbors as
#         :math:`\square_{j \in \mathcal{N}(i)}`.

#         Takes in the output of message computation as first argument and any
#         argument which was initially passed to :meth:`propagate`.

#         By default, this function will delegate its call to scatter functions
#         that support "add", "mean" and "max" operations as specified in
#         :meth:`__init__` by the :obj:`aggr` argument.
#         """
# #         ipdb.set_trace()
#         if aggr is None:
#             raise ValeuError("aggr was not passed!")
#         return scatter(inputs, index, dim=self.node_dim, reduce=aggr)
    



class MLP(nn.Module):
    """ adapted from https://github.com/CUAI/CorrectAndSmooth/blob/master/gen_models.py """

    def __init__(self, in_channels, hidden_channels, out_channels, num_layers,
                 dropout=.5, Normalization='bn', InputNorm=False):
        super(MLP, self).__init__()
        self.lins = nn.ModuleList()
        self.normalizations = nn.ModuleList()
        self.InputNorm = InputNorm
        
        if num_layers == 1:
            # just linear layer i.e. logistic regression
            if InputNorm:
                self.normalizations.append(nn.LayerNorm(in_channels))
            else:
                self.normalizations.append(nn.Identity())
            self.lins.append(nn.Linear(in_channels, out_channels))
        else:
            if InputNorm:
                self.normalizations.append(nn.LayerNorm(in_channels))
            else:
                self.normalizations.append(nn.Identity())
            self.lins.append(nn.Linear(in_channels, hidden_channels))
            self.normalizations.append(nn.LayerNorm(hidden_channels))
            for _ in range(num_layers - 2):
                self.lins.append(
                    nn.Linear(hidden_channels, hidden_channels))
                self.normalizations.append(nn.LayerNorm(hidden_channels))
            self.lins.append(nn.Linear(hidden_channels, out_channels))
        
        self.dropout = dropout

    def reset_parameters(self):
        for lin in self.lins:
            lin.reset_parameters()
        for normalization in self.normalizations:
            if not (normalization.__class__.__name__ is 'Identity'):
                normalization.reset_parameters()

    def forward(self, x):
        x = self.normalizations[0](x)
        for i, lin in enumerate(self.lins[:-1]):
            x = lin(x)
            x = F.relu(x, inplace=True)
            x = self.normalizations[i+1](x)
            x = F.dropout(x, p=self.dropout, training=self.training)
        x = self.lins[-1](x)
        return x





