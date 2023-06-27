"""Simplicial Attention Network (SAN) Layer."""
import torch
from torch.nn import functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SANConv(Conv):
    r"""Class for the SAN Convolution."""

    def __init__(
        self,
        in_channels,
        out_channels,
        p_filters,
        initialization="xavier_uniform",
    ):
        super(Conv, self).__init__(
            att=True,
            initialization=initialization,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.p_filters = p_filters
        self.initialization = initialization

        self.weight = Parameter(
            torch.Tensor(self.p_filters, self.in_channels, self.out_channels)
        )

        self.att_weight = Parameter(
            torch.Tensor(
                2 * self.out_channels * self.p_filters,
            )
        )

        self.reset_parameters()

    def forward(self, x_source, neighborhood):
        """Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells, which are the same source cells.

        In practice, this will update the features on the target cells.

        If not provided, x_target is assumed to be x_source,
        i.e. source cells send messages to themselves.

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        _ : Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        x_message = torch.matmul(x_source, self.weight)
        # Reshape required to re-use the attention function of parent Conv class
        # -> [num_nodes, out_channels * p_filters]
        x_message_reshaped = x_message.permute(1, 0, 2).reshape(
            -1, self.out_channels * self.p_filters
        )

        # SAN always requires attention
        # In SAN, neighborhood is defined by lower/upper laplacians; we only use them as masks
        # to keep only the relevant attention coeffs
        neighborhood = neighborhood.coalesce()
        self.target_index_i, self.source_index_j = neighborhood.indices()
        attention_values = self.attention(x_message_reshaped)
        att_laplacian = torch.sparse_coo_tensor(
            indices=neighborhood.indices(),
            values=attention_values,
            size=neighborhood.shape,
        )

        # Attention coeffs are normalized using softmax
        att_laplacian = torch.sparse.softmax(att_laplacian, dim=1).to_dense()
        # We need to compute the power of the attention laplacian according to the filter order p
        att_laplacian_power = torch.stack(
            [
                torch.linalg.matrix_power(att_laplacian, p + 1)
                if p > 1
                else att_laplacian
                for p in range(1, self.p_filters + 1)
            ]
        )

        # When computing the final message on targets, we need to compute the power of the attention laplacian
        # according to the filter order p
        x_message_on_target = torch.matmul(att_laplacian_power, x_message).sum(dim=0)

        return x_message_on_target


class SANLayer(torch.nn.Module):
    r"""Class for the SAN layer."""

    def __init__(
        self,
        in_channels,
        out_channels,
        num_filters_J=2,  # approximation order
        J_har=5,  # approximation order for harmonic
        epsilon_har=1e-1,  # epsilon for harmonic, it takes into account the normalization
    ):
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_filters_J = num_filters_J

        self.J_har = J_har
        self.epsilon_har = epsilon_har

        #  Convolutions
        # Down convolutions, one for each filter order p
        self.conv_down = SANConv(in_channels, out_channels, num_filters_J)

        # Up convolutions, one for each filter order p
        self.conv_up = SANConv(in_channels, out_channels, num_filters_J)

        # Harmonic convolution
        self.conv_har = Conv(in_channels, out_channels)

    def reset_parameters(self):
        r"""Reset learnable parameters."""
        # Following original repo.
        gain = torch.nn.init.calculate_gain("relu")
        for p in range(self.num_filters_J):
            torch.nn.init.xavier_uniform_(self.convs_down[p].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.convs_down[p].att_weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.convs_up[p].weight, gain=gain)
            torch.nn.init.xavier_uniform_(self.convs_up[p].att_weight, gain=gain)

    def forward(self, x, Lup, Ldown, P):
        r"""Forward pass.

        The forward pass was initially proposed in [HRGZ22]_.
        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.
        """
        # For the down and up convolutions, we sum the outputs for each filter order p
        z_down = self.conv_down(x, Ldown)
        z_up = self.conv_up(x, Lup)
        # For the harmonic convolution, we use the precomputed projection matrix P as the neighborhood
        # with no attention
        z_har = self.conv_har(x, P)

        # final output
        x = z_down + z_up + z_har
        return x
