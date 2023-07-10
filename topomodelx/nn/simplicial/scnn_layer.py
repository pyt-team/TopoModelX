import torch
from torch.nn.parameter import Parameter


class SCNNLayer(torch.nn.Module):
    """Layer of a Simplicial Convolutional Neural Network (SCNN).
    
    Notes
    -----
    This is Implementation of the SCNN layer.
    
    References
    ----------
    [Yang et. al : SIMPLICIAL CONVOLUTIONAL NEURAL NETWORKS (2022)]
    (https://arxiv.org/pdf/2110.02585.pdf)
    
    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    conv_order: int
      The order of the convolutions.
      if conv_order == 0:
        the corresponding convolution is not performed
      - down: for the lower convolutions
      - up: for the upper convolutions
    Example
    -------
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
    """

    def __init__(
        self,
        in_channels,
        out_channels,
        conv_order_down,
        conv_order_up,
        aggr_norm=False,
        update_func=None,
        initialization="xavier_uniform",
    ):
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

    def reset_parameters(self, gain=1.414):
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
        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.weight, gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def aggr_norm_func(self, conv_operator, x):
        r""" aggregation normalization
        """
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
        x_message_on_target: torch.Tensor, shape=[n_target_cells, out_channels]
            Output features on target cells.

        Returns
        -------
        _ : torch.Tensor, shape=[n_target_cells, out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(x)
        if self.update_func == "relu":
            return torch.nn.functional.relu(x)

    def chebyshev_conv(self, conv_operator, conv_order, x):
        r"""A Chebyshev convolution method.
        Parameters
        ----------
        conv_operator: torch.sparse
          shape = [n_simplices,n_simplices]
          e.g. adjacency matrix or the Hodge Laplacians
        conv_order: int
          the order of the convolution
        x : torch.Tensor
          shape = [n_simplices,num_channels]
          
        Return
        ------
          x[:,:,k] = (conv_operator@....@conv_operator) @ x
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
        r"""Forward computation.
    
        Parameters
        ----------
        x: torch.Tensor, shape=[n_simplex,in_channels]
          Inpute features on the simplices, e.g., nodes, edges, triangles, etc.

        laplacian: torch.sparse
          shape = [n_simplices,n_simplices]
          The Hodge Laplacian matrix
            - can also be adjacency matrix
            - lower part
            - upper part
            
        Returns
        -------
        _ : torch.Tensor, shape=[n_edges, channels]
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
        else:
            x = x_identity

        y = torch.einsum("nik,iok->no", x, self.weight)

        if self.update_func is None:
            return y

        return self.update(y)
