"""Simplicial Attention Network (SAN) Layer."""
import torch
from torch.nn import functional as F
from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SANLayer(torch.nn.Module):
    """

    """

    def __init__(
        self,
        channels_in,
        channels_out,
        J=2, # approximation order
    ):
        super().__init__()
        
        #self.Lup, self.Ld, self.P = Lup, Ld, P
        
        self.J = J
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.att_slice = self.channels_out * self.J
        
        # Cell Convolution

        self.W_irr =  [
                        torch.nn.Linear(in_features=self.channels_in,
                                        out_features=self.channels_out)
                        for _ in range(self.J)
                        ]

        self.W_sol = [
                        torch.nn.Linear(in_features=self.channels_in,
                                        out_features=self.channels_out)
                            for _ in range(self.J)
                          ]
        
        self.W_har = torch.nn.Linear(in_features=self.channels_in,
                                    out_features=self.channels_out)
        
        # Attention
        self.att_irr = torch.nn.Parameter(torch.empty(size=(2 * self.att_slice, 1)))
        self.att_sol = torch.nn.Parameter(torch.empty(size=(2 * self.att_slice, 1)))

        # Summation
        self.aggr_on_nodes = Aggregation(aggr_func="sum", update_func=None)

        self.bin_mask_Lu = None
        self.bin_mask_Ld = None

    # def reset_parameters(self):
    #     r"""Reset learnable parameters."""
    #     # Following original repo.
    #     gain = torch.nn.init.calculate_gain('relu')
    #     torch.nn.init.xavier_uniform_(self.conv_down.weight, gain=gain)
    #     torch.nn.init.xavier_uniform_(self.conv_up.weight, gain=gain)
        
        

    def forward(self, x, Lup, Ld, P):
        r"""Forward pass.

        The forward pass was initially proposed in [HRGZ22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        .. math::
           

        References
        ----------
        

        Parameters
        ----------
        
        Returns
        -------
        
        """
        
        x_irr = torch.cat([self.W_irr[i](x) for i in range(self.J)], dim=1)
        x_sol = torch.cat([self.W_sol[i](x) for i in range(self.J)], dim=1)


        # Broadcast add
        # Attention map
        
        # (Ex1) + (1xE) -> (ExE)
        E_irr = F.leaky_relu((x_irr @ self.att_irr[:self.att_slice, :]) + 
                                (x_irr @ self.att_irr[self.att_slice:, :]).T) 
        
        # (Ex1) + (1xE) -> (ExE) 
        E_sol = F.leaky_relu((x_sol @ self.att_sol[:self.att_slice, :]) + 
                                (x_sol @ self.att_sol[self.att_slice:, :]).T)  
        

        # Consider only ones which have connections
        MASK_D, MASK_U = Ld != 0, Lup != 0
        E_irr[~MASK_D] = float("-1e20")
        E_sol[~MASK_U] = float("-1e20")
        
       

        # Optional dropout
        # (ExE) -> (ExE)
        alpha_irr = F.softmax(E_irr, dim=-1)
                               
        # (ExE) -> (ExE)
        alpha_sol = F.softmax(E_sol, dim=-1) 


        
        z_i = self.W_irr[0](torch.matmul(alpha_irr, x))
        z_s = self.W_sol[0](torch.matmul(alpha_sol, x))  
        for p in range(1, self.J):
            alpha_irr = torch.matmul(alpha_irr, Ld)
            alpha_sol = torch.matmul(alpha_sol, Lup)
            
            # (ExE) x (ExF_out) -> (ExF_out)
            z_i = z_i + self.W_irr[p](torch.matmul(alpha_irr, x))
            # (ExE) x (ExF_out) -> (ExF_out)
            z_s = z_s + self.W_sol[p](torch.matmul(alpha_sol, x))  

        
        # Harmonic
        z_har = self.W_har(torch.matmul(P, x))
        
        # final output
        x = (z_i + z_s + z_har)
        return x
