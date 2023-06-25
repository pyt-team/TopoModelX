from typing import Optional, Tuple, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Parameter

in_channels = 3
out_channels = C = 8

nodes = N = 5
heads = H = 2

concat = True

x = torch.randn(nodes, in_channels)
neighborhood = (torch.randn(nodes, nodes)).fill_diagonal_(0).to_sparse().float()

target_index_i, source_index_j = neighborhood.indices()

lin = torch.nn.Linear(in_channels, heads * out_channels, bias=False)
alpha_src = Parameter(torch.Tensor(1, heads, out_channels))
alpha_dst = Parameter(torch.Tensor(1, heads, out_channels))

x_transformed = lin(x).view(-1, H, C) # (N, H, C)

x_source_per_message = x_transformed[source_index_j] # (E, H, C)
print("x_source_per_message", x_source_per_message.shape)
x_target_per_message = x_transformed[target_index_i] # (E, H, C)

alpha_src = (x_source_per_message * alpha_src).sum(dim=-1) # (E, H)
alpha_dst = (x_target_per_message * alpha_dst).sum(dim=-1) # (E, H)

# TODO: apply a leaky relu on the attention coefficients
alpha_src = F.leaky_relu(alpha_src, 0.01) # (E, H)
alpha_dst = F.leaky_relu(alpha_dst, 0.01) # (E, H)

# TODO: for each head, updates the neighborhood with the attention coefficients
neighborhood_values = neighborhood.values()
alpha = alpha_src + alpha_dst
updated_neighborhood = neighborhood_values[:,None] + alpha # Broadcasting addition

# TODO: normalize the neighborhood for each head with the softmax function applied on rows of the neighborhood
normalized_neighborhood = F.softmax(updated_neighborhood, dim=1) # (E, H)

# TODO: for each head, Aggregate the messages
message = x_source_per_message * normalized_neighborhood[:,:,None] # (E, H, C)
out = torch.zeros(N, H, C, device=x.device)
out.index_add_(0, target_index_i, message)

# TODO: if concat true, concatenate the messages for each head. Otherwise, average the messages for each head.
if concat:
    out = out.view(-1, heads * out_channels)
else:
    out = out.mean(dim=1)
print("out.shape --> ", out.shape)