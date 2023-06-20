"""Simplicial Attention Network (SAN) Layer."""
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SANLayer(torch.nn.Module):
    r"""Class for the SAN layer."""

    def __init__(
        self,
        channels_in,
        channels_out,
        J=2,  # approximation order
        J_har=5,  # approximation order for harmonic
        epsilon_har=1e-1,  # epsilon for harmonic, it takes into account the normalization
    ):
        super().__init__()

        self.J = J
        self.J_har = J_har
        self.epsilon_har = epsilon_har
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.att_slice = self.channels_out * self.J

        #  Convolution
        self.weight_irr = Parameter(
            torch.Tensor(self.J, self.channels_in, self.channels_out)
        )
        self.weight_sol = Parameter(
            torch.Tensor(self.J, self.channels_in, self.channels_out)
        )
        self.weight_har = Parameter(torch.Tensor(self.channels_in, self.channels_out))
        # self.W_irr =  [
        #                torch.nn.Linear(in_features=self.channels_in,
        #                                out_features=self.channels_out)
        #                for _ in range(self.J)
        #                ]

        # self.W_sol = [
        #                torch.nn.Linear(in_features=self.channels_in,
        #                                out_features=self.channels_out)
        #                    for _ in range(self.J)
        #                  ]

        self.W_har = torch.nn.Linear(
            in_features=self.channels_in, out_features=self.channels_out
        )

        # Attention
        self.att_irr = Parameter(torch.Tensor(2 * self.att_slice, 1))
        self.att_sol = Parameter(torch.Tensor(2 * self.att_slice, 1))

        # self.att_irr = torch.nn.Parameter(torch.empty(size=(2 * self.att_slice, 1)))
        # self.att_sol = torch.nn.Parameter(torch.empty(size=(2 * self.att_slice, 1)))

        # self.bin_mask_Lu = None
        # self.bin_mask_Ld = None

    # def reset_parameters(self):
    #     r"""Reset learnable parameters."""
    #     # Following original repo.
    #     gain = torch.nn.init.calculate_gain('relu')
    #     torch.nn.init.xavier_uniform_(self.conv_down.weight, gain=gain)
    #     torch.nn.init.xavier_uniform_(self.conv_up.weight, gain=gain)

    def forward(self, x, Lup, Ldown, P):
        r"""Forward pass.

        The forward pass was initially proposed in [HRGZ22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.
        """
        h_irr = torch.matmul(x, self.weight_irr)
        h_sol = torch.matmul(x, self.weight_sol)
        # x_irr = torch.cat([torch.mm(x,self.weight_irr[i]) for i in range(self.J)], dim=1)
        # x_sol = torch.cat([torch.mm(x,self.weight_sol[i]) for i in range(self.J)], dim=1)

        # Broadcast add
        # Attention map

        # (Ex1) + (1xE) -> (ExE)
        e_irr = F.leaky_relu(
            torch.mm(
                h_irr.reshape(-1, self.J * self.channels_out),
                self.att_irr[: self.att_slice, :],
            )
            + torch.mm(
                h_irr.reshape(-1, self.J * self.channels_out),
                self.att_irr[self.att_slice :, :],
            ).T
        )
        e_sol = F.leaky_relu(
            torch.mm(
                h_sol.reshape(-1, self.J * self.channels_out),
                self.att_sol[: self.att_slice, :],
            )
            + torch.mm(
                h_sol.reshape(-1, self.J * self.channels_out),
                self.att_sol[self.att_slice :, :],
            ).T
        )

        # Consider only ones which have connections
        alpha_irr = torch.sparse.softmax(e_irr.sparse_mask(Ldown), dim=1).to_dense()
        alpha_sol = torch.sparse.softmax(e_sol.sparse_mask(Lup), dim=1).to_dense()
        # MASK_D, MASK_U = Ldown != 0, Lup != 0
        # e_irr[~MASK_D] = float("-1e20")
        # e_sol[~MASK_U] = float("-1e20")

        # Optional dropout
        # (ExE) -> (ExE)
        # alpha_irr = F.softmax(e_irr, dim=-1)
        # alpha_sol = F.softmax(e_sol, dim=-1)

        alpha_exp_irr = alpha_irr.unsqueeze(0)
        alpha_exp_sol = alpha_sol.unsqueeze(0)
        for p in range(self.J - 1):
            alpha_exp_irr = torch.cat(
                [alpha_exp_irr, torch.mm(alpha_exp_irr[p], alpha_irr).unsqueeze(0)],
                dim=0,
            )
            alpha_exp_sol = torch.cat(
                [alpha_exp_sol, torch.mm(alpha_exp_sol[p], alpha_sol).unsqueeze(0)],
                dim=0,
            )
        z_irr = torch.sum(torch.matmul(alpha_exp_irr.to_dense(), h_irr))
        z_sol = torch.sum(torch.matmul(alpha_exp_sol.to_dense(), h_sol))

        # WRONG!! Alphas should be multiplied by themselves, not by Laplacians!!!!
        # z_i = torch.mm(torch.matmul(alpha_irr, x),self.weight_irr[0])
        # z_s = torch.mm(torch.matmul(alpha_sol, x),self.weight_sol[0])
        # for p in range(1, self.J):
        #    alpha_irr = torch.matmul(alpha_irr, Ldown)
        #    alpha_sol = torch.matmul(alpha_sol, Lup)

        # (ExE) x (ExF_out) -> (ExF_out)
        #    z_i = z_i + torch.mm(torch.matmul(alpha_irr, x),self.weight_irr[p])
        # (ExE) x (ExF_out) -> (ExF_out)
        #    z_s = z_s + torch.mm(torch.matmul(alpha_sol, x),self.weight_sol[p])

        # Harmonic component
        z_har = torch.mm(torch.matmul(P, x), self.weight_har)

        # final output
        x = z_irr + z_sol + z_har
        return x
