"""Simplicial Convolutional Neural Network Layer."""
import torch
from torch.nn.parameter import Parameter


class SCNNLayer(torch.nn.Module):
    r"""Layer of a Simplicial Convolutional Neural Network (SCNN) [1]_.

    Notes
    -----
    This is Implementation of the SCNN layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    conv_order : int
        The order of the convolutions.
        if conv_order == 0:
            the corresponding convolution is not performed.
        - down: for the lower convolutions.
        - up: for the upper convolutions.

    Examples
    --------
    Here we provide an example of pseudocode for SCNN layer
    input X: [n_simplices, in_channels]
    Lap_down, Lap_up: [n_simplices, n_simplices]
    conv_order_down: int, e.g., 2
    conv_order_up: int, e.g., 2
    output Y: [n_simplices, out_channels]

    SCNN layer looks like:

      Y = torch.einsum(concat(X, Lap_down@X, Lap_down@Lap_down@X, Lap_up@X,
                              Lap_up@Lap_up@X), weight)
    where
      - weight is the trainable parameters of dimension
            [out_channels,in_channels, total_order]
      - total_order = 1 + conv_order_down + conv_order_up
      - to implement Lap_down@Lap_down@X, we consider chebyshev
        method to avoid matrix@matrix computation

    References
    ----------
    .. [1] Yang, Isufi and Leus.
        Simplicial Convolutional Neural Networks (2021).
        https://arxiv.org/pdf/2110.02585.pdf
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        aggr_norm: bool = False,
        update_func=None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.conv_order_down = conv_order_down
        self.conv_order_up = conv_order_up
        self.aggr_norm = aggr_norm
        self.update_func = update_func
        self.initialization = initialization
        assert initialization in ["xavier_uniform", "xavier_normal"]

        self.weight = Parameter(
            torch.Tensor(
                self.in_channels,
                self.out_channels,
                1 + self.conv_order_down + self.conv_order_up,
            )
        )

        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414) -> None:
        r"""Reset learnable parameters.

        Notes
        -----
        This function will be called by subclasses of
        MessagePassing that have trainable weights.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.weight, gain=gain)

    def aggr_norm_func(self, conv_operator, x):
        r"""Perform aggregation normalization."""
        neighborhood_size = torch.sum(conv_operator.to_dense(), dim=1)
        neighborhood_size_inv = 1 / neighborhood_size
        neighborhood_size_inv[~(torch.isfinite(neighborhood_size_inv))] = 0

        x = torch.einsum("i,ij->ij ", neighborhood_size_inv, x)
        x[~torch.isfinite(x)] = 0
        return x

    def update(self, x):
        """Update embeddings on each cell (step 4).

        Parameters
        ----------
        x : torch.Tensor, shape = (n_target_cells, out_channels)
            Output features on target cells.

        Returns
        -------
        torch.Tensor, shape = (n_target_cells, out_channels)
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        elif self.update_func == "relu":
            return torch.nn.functional.relu(x)

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""Perform Chebyshev convolution.

        Parameters
        ----------
        conv_operator : torch.sparse, shape = (n_simplices,n_simplices)
            Convolution operator e.g. adjacency matrix or the Hodge Laplacians.
        conv_order : int
            The order of the convolution
        x : torch.Tensor, shape = (n_simplices,num_channels)
            Input feature tensor.

        Return
        ------
        torch.Tensor
            Output tensor, x[:,:,k] = (conv_operator@....@conv_operator) @ x.
        """
        num_simplices, num_channels = x.shape
        X = torch.empty(size=(num_simplices, num_channels, conv_order))
        X[:, :, 0] = torch.mm(conv_operator, x)
        for k in range(1, conv_order):
            X[:, :, k] = torch.mm(conv_operator, X[:, :, k - 1])
            if self.aggr_norm:
                X[:, :, k] = self.aggr_norm_func(conv_operator, X[:, :, k])

        return X

    def forward(self, x, laplacian_down, laplacian_up):
        r"""Forward computation ([2]_ and [3]_).

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{p,u,(1 \rightarrow 2 \rightarrow 1)}  = ((L_{\uparrow,1})^u)\_{xy} \cdot h_y^{t,(1)} \cdot (\alpha^{t, p, u} \cdot I)\\
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{p,d,(1 \rightarrow 0 \rightarrow 1)} = ((L_{\downarrow,1})^d)\_{xy} \cdot h_y^{t,(1)} \cdot (\alpha^{t, p, d} \cdot I)\\
            &🟥 \quad m^{(1 \rightarrow 1)}\_{x \rightarrow x} = \alpha \cdot h_x^{t, (1)}\\
            &🟧 \quad m_{x}^{p,u,(1 \rightarrow 2 \rightarrow 1)}  = \sum_{y \in \mathcal{L}\_\uparrow(X)}m_{y \rightarrow \{z\} \rightarrow x}^{p,u,(1 \rightarrow 2 \rightarrow 1)}\\
            &🟧 \quad m_{x}^{p,d,(1 \rightarrow 0 \rightarrow 1)} = \sum_{y \in \mathcal{L}\_\downarrow(X)}m_{y \rightarrow \{z\} \rightarrow x}^{p,d,(1 \rightarrow 0 \rightarrow 1)}\\
            &🟧 \quad m^{(1 \rightarrow 1)}\_{x} = m^{(1 \rightarrow 1)}\_{x \rightarrow x}\\
            &🟩 \quad m_x^{(1)}  = m_x^{(1 \rightarrow 1)} + \sum_{p=1}^P( \sum_{u=1}^{U} m_{x}^{p,u,(1 \rightarrow 2 \rightarrow 1)} + \sum_{d=1}^{D} m_{x}^{p,d,(1 \rightarrow 0 \rightarrow 1)})\\
            &🟦 \quad h_x^{t+1, (1)} = \sigma(m_x^{(1)})
            \end{align*}

        Parameters
        ----------
        x: torch.Tensor, shape = (n_simplex,in_channels)
            Input features on the simplices, e.g., nodes, edges, triangles, etc.

        laplacian: torch.sparse, shape = (n_simplices,n_simplices)
            The Hodge Laplacian matrix. Can also be adjacency matrix, lower part, or upper part.

        Returns
        -------
        torch.Tensor, shape = (n_edges, channels)
            Output features on the edges of the simplical complex.
        """
        num_simplices, _ = x.shape

        identity = torch.eye(num_simplices)

        x_identity = torch.unsqueeze(identity @ x, 2)

        if self.conv_order_down > 0 and self.conv_order_up > 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_down, x_up), 2)
        elif self.conv_order_down > 0 and self.conv_order_up == 0:
            x_down = self.chebyshev_conv(laplacian_down, self.conv_order_down, x)
            x = torch.cat((x_identity, x_down), 2)
        elif self.conv_order_down == 0 and self.conv_order_up > 0:
            x_up = self.chebyshev_conv(laplacian_up, self.conv_order_up, x)
            x = torch.cat((x_identity, x_up), 2)

        y = torch.einsum("nik,iok->no", x, self.weight)

        if self.update_func is None:
            return y

        return self.update(y)
