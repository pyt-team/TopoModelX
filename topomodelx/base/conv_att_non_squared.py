"""Convolutional layer for message passing."""

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


class ConvNonSquared:
    def __init__(
            self,
            in_channels_source,
            in_channels_target,
            out_channels_source,
            out_channels_target,
            update_func=None,
            initialization="xavier_uniform",
    ):
        super().__init__(
            initialization=initialization,
        )
        self.in_channels_source = in_channels_source
        self.in_channels_target = in_channels_target
        self.out_channels_source = out_channels_source
        self.out_channels_target = out_channels_target

        # We have two weight matrices W_s and W_t. W_s is applied to the source cells, W_t to the target cells.
        # W_s is of shape [in_channels_source, out_channels_target]
        # W_t is of shape [in_channels_target, out_channels_source]
        self.W_s = Parameter(torch.Tensor(self.in_channels_source, self.out_channels_target))
        self.W_t = Parameter(torch.Tensor(self.in_channels_target, self.out_channels_source))

        # The attention weights are a vector of dimension [out_channels_source + out_channels_target]
        self.att_weight = Parameter(torch.Tensor(self.out_channels_source + self.out_channels_target))

        # Non-linearity function for message passing
        self.update_func = update_func

    def concat_attentions(self, att_left_side, att_right_side):
        rows_left_side, columns_left_side = att_left_side.shape
        rows_right_side, columns_right_side = att_right_side.shape
        # Expand the dimensions of A and B
        att_left_side_expanded = att_left_side.unsqueeze(1)  # Shape: [rows_left_side, 1, columns_left_side]
        att_right_side_expanded = att_right_side.unsqueeze(0)  # Shape: [1, rows_right_side, columns_right_side]

        # Repeat att_left_side and att_right_side along the expanded dimensions
        att_left_side_repeated = att_left_side_expanded.repeat(1, rows_right_side,
                                                               1)  # Shape: [rows_left_side, rows_right_side, columns_left_side]
        att_right_side_repeated = att_right_side_expanded.repeat(rows_left_side, 1,
                                                                 1)  # Shape: [rows_left_side, rows_right_side, columns_right_side]

        # Concatenate A_repeated and B_repeated along the last dimension
        attention = torch.cat((att_left_side_repeated, att_right_side_repeated),
                              dim=2)  # Shape: [rows_left_side, rows_right_side, columns_left_side + columns_right_side]
        return attention

    @staticmethod
    def compute_attention_coefficients(attention_concatenated, att_weights):
        return torch.einsum('ijk,k->ij', attention_concatenated, att_weights)

    def compute_attention_matrices(self, x_source, neighborhood, x_target):
        # x_source = [n_source, in_channels_source]
        # x_target = [n_target, in_channels_target]
        att_left_side = torch.mm(x_source, self.W_s)  # Shape: [n_source, out_channels_target]
        att_right_side = torch.mm(x_target, self.W_t)  # Shape: [n_target, out_channels_source]
        attention_s_to_t = self.concat_attentions(att_left_side, att_right_side)
        attention_t_to_s = self.concat_attentions(att_right_side, att_left_side)
        # Multiply the attention weights with the attention matrix
        att_s_to_t = self.compute_attention_coefficients(attention_s_to_t, self.att_weight)
        att_weights_t_to_s = torch.cat(self.att_weight[self.out_channels_target:],
                                       self.att_weight[:self.out_channels_target])
        att_t_to_s = self.compute_attention_coefficients(attention_t_to_s, att_weights_t_to_s)
        # Multiply element-wise the attention matrix with the neighborhood matrix
        att_s_to_t_neigh = torch.mul(neighborhood, att_t_to_s)
        att_t_to_s_neigh = torch.mul(neighborhood, att_s_to_t)
        # Normalize the attention matrix, it is normalized by rows by dividing by the sum of the row
        att_s_to_t_final = att_s_to_t_neigh / (att_s_to_t_neigh.sum(dim=1)[:, None])
        att_t_to_s_final = att_t_to_s_neigh / (att_t_to_s_neigh.sum(dim=1)[:, None])
        return att_s_to_t_final, att_t_to_s_final

    def forward(self, x_source, neighborhood, x_target):
        att_s_to_t, att_t_to_s = self.compute_attention_matrices(x_source, neighborhood, x_target)
        # Compute the messages
        messages_s_to_t = torch.mm(torch.mm(att_s_to_t, x_source), self.W_s)
        messages_t_to_s = torch.mm(torch.mm(att_t_to_s, x_target), self.W_t)
        # Update the messages
        if self.update_func is None:
            return messages_s_to_t, messages_t_to_s
        else:
            return self.update_func(messages_s_to_t), self.update_func(messages_t_to_s)
