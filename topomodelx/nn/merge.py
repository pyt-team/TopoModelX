

from typing import Optional, Tuple

import torch
from topomodelx.scatter.scatter import scatter
from topomodelx.nn.message_passing import HigherOrderMessagePassing
from topomodelx.nn.linear import Linear

from torch import Tensor



class Merge(HigherOrderMessagePassing):
    """
    from toponetx import SimplicialComplex
    from topomodelx.util.tensors_util import coo_2_torch_tensor
    from topomodelx.nn.message_passing import HigherOrderMessagePassing
    SC= SimplicialComplex([[0,1],[1,2]])
    B1 = coo_2_torch_tensor(SC.incidence_matrix(1))
    A0 = coo_2_torch_tensor(SC.adjacency_matrix(0))
    n_v,n_e = B1.shape
    merge = Merge(10,8,16)
    
    x_e = torch.rand(n_e,10)
    x_v = torch.rand(n_v,8)
    
    x_v_out=merge(x_e,x_v,B1,A0)
    
    
    """
    
    def __init__(self,in_ch_1,
                    in_ch_2,
                    target_ch, merge="conc"): 

        super().__init__()
        self.merge = merge
        self.linear1 = Linear(in_ch_1, target_ch)
        self.linear2 = Linear(in_ch_2, target_ch)
       
    
    def forward(self,x1,x2,G1,G2):
        out1 = self.propagate(self.linear1(x1),G1)
        out2 = self.propagate(self.linear2(x2),G2)
        
        if self.merge == 'conc':
            return torch.cat((out1,out2))
        elif self.merge == 'sum':
            return out1+out2
            


        
