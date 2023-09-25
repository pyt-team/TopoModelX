"""Simplicial Attention Network (SAN) Layer."""
from typing import Literal

import torch
from torch.nn.parameter import Parameter

from topomodelx.base.conv import Conv


class SANConv(Conv):
    r"""Simplicial Attention Network (SAN) Convolution from [1]_.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_filters : int
        Number of simplicial filters.
    initialization : Literal["xavier_uniform", "xavier_normal"], default="xavier_uniform"
        Weight initialization method.

    References
    ----------
    .. [1] Giusti, Battiloro, Di Lorenzo, Sardellitti and Barbarossa.
        Simplicial attention neural networks (2022).
        https://arxiv.org/abs/2203.07485.
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_filters,
        initialization: Literal["xavier_uniform", "xavier_normal"] = "xavier_uniform",
    ) -> None:
        super(Conv, self).__init__(
            att=True,
            initialization=initialization,
        )
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters
        self.initialization = initialization

        self.weight = Parameter(
            torch.Tensor(self.n_filters, self.in_channels, self.out_channels)
        )

        self.att_weight = Parameter(
            torch.Tensor(
                2 * self.out_channels * self.n_filters,
            )
        )

        self.reset_parameters()

    def forward(self, x_source, neighborhood):
        r"""Forward pass.

        This implements message passing:
        - from source cells with input features `x_source`,
        - via `neighborhood` defining where messages can pass,
        - to target cells, which are the same source cells.

        In practice, this will update the features on the target cells [2]_.

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{u,(1 \rightarrow 2 \rightarrow 1)}  = ((L_{\uparrow,1} \odot \operatorname{att}(h_z^{t,(2)}, h_y^{t,(1)}))^u)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,u}\\
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{d,(1 \rightarrow 0 \rightarrow 1)}  = ((L_{\downarrow,1} \odot \operatorname{att}(h_z^{t,(0)}, h_y^{t,(1)}))^d)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,d}\\
            &🟥 \quad m^{p,(1 \rightarrow 1)}\_{y \rightarrow x} = ((1-wH_1)^p)\_{xy} \cdot h_y^{t,(1)} \cdot \Theta^{t,p}\\
            &🟧 \quad m_{x}^{u,(1 \rightarrow 2 \rightarrow 1)}  = \sum_{y \in \mathcal{L}\_\uparrow(x)} m_{y \rightarrow \{z\} \rightarrow x}^{u,(1 \rightarrow 2 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{d,(1 \rightarrow 0 \rightarrow 1)}  = \sum_{y \in \mathcal{L}\downarrow(X)} m_{y \rightarrow \{z\} \rightarrow x}^{d,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟧 \quad m^{p,(1 \rightarrow 1)}\_{x}  = m^{p,(1 \rightarrow 1)}\_{x \rightarrow x}\\
            &🟩 \quad m_x^{(1)}  = \sum_{p=1}^P m_x^{p,(1 \rightarrow 1)} + \sum_{u=1}^{U} m_{x}^{u,(1 \rightarrow 2 \rightarrow 1)} + \sum_{d=1}^{D} m_{x}^{d,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟦 \quad h_x^{t+1, (1)} = \sigma(m_x^{(1)})
            \end{align*}

        Parameters
        ----------
        x_source : Tensor, shape=[..., n_source_cells, in_channels]
            Input features on source cells.
            Assumes that all source cells have the same rank r.
        neighborhood : torch.sparse, shape=[n_target_cells, n_source_cells]
            Neighborhood matrix.

        Returns
        -------
        torch.Tensor, shape=[..., n_target_cells, out_channels]
            Output features on target cells.
            Assumes that all target cells have the same rank s.
        """
        x_message = torch.matmul(x_source, self.weight)
        # Reshape required to re-use the attention function of parent Conv class
        # -> [num_nodes, out_channels * n_filters]
        x_message_reshaped = x_message.permute(1, 0, 2).reshape(
            -1, self.out_channels * self.n_filters
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
        # We need to compute the power of the attention laplacian according up to order p
        att_laplacian_power = [att_laplacian]
        for _ in range(1, self.n_filters):
            att_laplacian_power.append(
                torch.matmul(att_laplacian_power[-1], att_laplacian)
            )
        att_laplacian_power = torch.stack(att_laplacian_power)

        # When computing the final message on targets, we multiply the message by each power
        # of the attention laplacian and sum the results
        x_message_on_target = torch.matmul(att_laplacian_power, x_message).sum(dim=0)

        return x_message_on_target


class SANLayer(torch.nn.Module):
    r"""Implementation of the Simplicial Attention Network (SAN) Layer proposed in [1]_.

    Notes
    -----
    Architecture proposed for r-simplex (r>0) classification on simplicial complices.

    Parameters
    ----------
    in_channels : int
        Number of input channels.
    out_channels : int
        Number of output channels.
    n_filters : int, default = 2
        Approximation order.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        n_filters: int = 2,
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.n_filters = n_filters

        #  Convolutions
        # Down convolutions, one for each filter order p
        self.conv_down = SANConv(in_channels, out_channels, n_filters)

        # Up convolutions, one for each filter order p
        self.conv_up = SANConv(in_channels, out_channels, n_filters)

        # Harmonic convolution
        self.conv_harmonic = Conv(in_channels, out_channels)

    def reset_parameters(self) -> None:
        r"""Reset learnable parameters."""
        self.conv_down.reset_parameters()
        self.conv_up.reset_parameters()
        self.conv_harmonic.reset_parameters()

    def forward(self, x, laplacian_up, laplacian_down, projection_mat):
        r"""Forward pass of the SAN Layer.

        .. math::
            \mathcal N = \{\mathcal N_1, \mathcal N_2,...,\mathcal N_{2p+1}\}
                = \{A_{\uparrow, r}, A_{\downarrow, r}, A_{\uparrow, r}^2, A_{\downarrow, r}^2,...,A_{\uparrow, r}^p, A_{\downarrow, r}^p, Q_r\},

        .. math::
            \begin{align*}
            &🟥\quad m_{(y \rightarrow x),k}^{(r)}
                = \alpha_k(h_x^t,h_y^t) = a_k(h_x^{t}, h_y^{t}) \cdot \psi_k^t(h_x^{t})\quad \forall \mathcal N_k \in \mathcal{N}\\
            &🟧\quad m_{x,k}^{(r)}
                = \bigoplus_{y \in \mathcal{N}_k(x)}  m^{(r)}_{(y \rightarrow x),k}\\
            &🟩\quad m_{x}^{(r)}
                = \bigotimes_{\mathcal{N}_k\in\mathcal N}m_{x,k}^{(r)}\\
            &🟦\quad h_x^{t+1,(r)}
                = \phi^{t}(h_x^t, m_{x}^{(r)})
            \end{align*}

        Parameters
        ----------
        x : torch.Tensor, shape = (..., n_cells, in_channels)
            Input tensor.
        down_indices : torch.Tensor, shape = (..., n_cells_down, n_neighbors)
            Down indices tensor.
        up_indices : torch.Tensor, shape = (..., n_cells_up, n_neighbors)
            Up indices tensor.
        laplacians : tuple of torch.Tensor
            Tuple of lower and upper laplacians.

        Returns
        -------
        torch.Tensor, shape = (..., n_cells, out_channels)
            Output tensor.
        """
        # Compute the down and up convolutions
        z_down = self.conv_down(x, laplacian_down)
        z_up = self.conv_up(x, laplacian_up)
        # For the harmonic convolution, we use the precomputed projection matrix P as the neighborhood
        # with no attention
        z_har = self.conv_harmonic(x, projection_mat)

        # final output
        x = z_down + z_up + z_har
        return x
