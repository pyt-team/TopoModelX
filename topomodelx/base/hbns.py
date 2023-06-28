"""Higher Order Attention Block for non-squared neighborhoods (HBNS) for message passing module."""

import torch
import torch.nn.functional as F
from scipy.sparse import coo_matrix
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


# TODO : This should be in a utils file. We keep it here for now to present the code to the challenge.
def sparse_row_norm(sparse_tensor):
    """Normalize a sparse tensor by row dividing each row by its sum.

    Parameters
    ----------
    sparse_tensor : torch.sparse, shape=[n_cells, n_cells]

    Returns
    -------
    _ : torch.sparse, shape=[n_cells, n_cells]
        Normalized by rows sparse tensor.
    """
    row_sum = torch.sparse.sum(sparse_tensor, dim=1)
    values = sparse_tensor._values() / row_sum.to_dense()[sparse_tensor._indices()[0]]
    sparse_tensor = torch.sparse_coo_tensor(
        sparse_tensor._indices(), values, sparse_tensor.shape
    )
    return sparse_tensor.coalesce()


class HBNS(MessagePassing):
    r"""Higher Order Attention Block layer for non-squared neighborhoods (HBNS).

    This is a sparse implementation of an HBNS layer. HBNS layers were introduced in [HAJIJ23]_, Definition 31 and 33.

    Mathematically, a higher order attention block layer for non-squared neighborhood matrices N of shape
    [target_cells, source_cells] is a function that outputs new signals for the source and target cells of the
    application given by the neighborhood matrix N given input source and target cochain matrices Xs and Xt of shape
    [source_cells, source_in_channels] and [target_cells, target_in_channels], respectively. The output source and target
    cochain matrices Ys and Yt are computed as
     ..  math::
        \begin{align}
            Ys &= \phi((N^T \odot A_t) Xt W_t)\\
            Yt &= \phi((N \odot A_s) Xs W_s )
        \end{align}
    where the first product is the Hadamard product, and the other products are the usual matrix multiplication, W_t
    and W_s are learnable weight matrices of shapes [target_in_channels, target_out_channels] and
    [source_in_channels, source_out_channels], respectively, and A_t and A_s are attention matrices with the same shape
    of N^T and N, respectively. The entries (i, j) of the attention matrices A_t and A_s are computed as
    ..  math::
        \begin{align}
            A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{columns(N)} e_{i,k}} \\
            A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{columns(N^T)} f_{i,k}}
        \end{align}
    where
    ..  math::
        \begin{align}
            e_{i,j} &= S(\text{LeakyReLU}([Xs_jW_s||Xt_iW_t]a))\\
            f_{i,j} &= S(\text{LeakyReLU}([Xt_jW_t||Xs_iW_s][a[source_out_channels:]||a[:source_out_channels]]))\\
        \end{align}
    and where || denotes concatenation, a is a learnable column vector of length
    source_out_channels + target_out_channels, v[:c] is the vector consisting of the first c elements of v, v[c:]
    is the vector consisting of the last elements of v, starting from the (c+1)-th element, and S is the exponential
    function if softmax is used and the identity function otherwise.

    References
    ----------
    .. [HAJIJ23] Mustafa Hajij et al. Topological Deep Learning: Going Beyond Graph Data.
        arXiv:2206.00606.
        https://arxiv.org/pdf/2206.00606v3.pdf

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    target_in_channels : int
        Number of input features for the target cells.
    target_out_channels : int
        Number of output features for the target cells.
    negative_slope : float
        Negative slope of the LeakyReLU activation function.
    softmax : bool, optional
        Whether to use softmax in the computation of the attention matrix. Default is False.
    update_func : {None, 'sigmoid', 'relu'}, optional
        phi function in the computation of the output of the layer. If None, phi is the identity function. Default is
        None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of W_p and a. Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        source_in_channels,
        source_out_channels,
        target_in_channels,
        target_out_channels,
        negative_slope=0.2,
        softmax=False,
        update_func=None,
        initialization="xavier_uniform",
    ):
        super().__init__(
            att=True,
            initialization=initialization,
        )

        self.source_in_channels, self.source_out_channels = (
            source_in_channels,
            source_out_channels,
        )
        self.target_in_channels, self.target_out_channels = (
            target_in_channels,
            target_out_channels,
        )

        self.update_func = update_func

        self.w_s = Parameter(
            torch.Tensor(self.source_in_channels, self.target_out_channels)
        )
        self.w_t = Parameter(
            torch.Tensor(self.target_in_channels, self.source_out_channels)
        )

        self.att_weight = Parameter(
            torch.Tensor(self.target_out_channels + self.source_out_channels, 1)
        )
        self.negative_slope = negative_slope

        self.softmax = softmax

        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization. Default is 1.414.
        """
        if self.initialization == "xavier_uniform":
            torch.nn.init.xavier_uniform_(self.w_s, gain=gain)
            torch.nn.init.xavier_uniform_(self.w_t, gain=gain)
            torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)

        elif self.initialization == "xavier_normal":
            torch.nn.init.xavier_normal_(self.w_s, gain=gain)
            torch.nn.init.xavier_normal_(self.w_t, gain=gain)
            torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
        else:
            raise RuntimeError(
                "Initialization method not recognized. "
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(self, message_on_source, message_on_target):
        """Update embeddings on each cell.

        Parameters
        ----------
        message_on_source : torch.Tensor, shape=[n_cells, out_channels]
            Output features of the layer for the source before the update function.
        message_on_target : torch.Tensor, shape=[n_cells, out_channels]
            Output features of the layer for the target before the update function.

        Returns
        -------
        _ : torch.Tensor, shape=[source_cells, source_out_channels]
            Updated output features on source cells.
        _ : torch.Tensor, shape=[target_cells, target_out_channels]
            Updated output features on target cells.
        """
        if self.update_func == "sigmoid":
            message_on_source = torch.sigmoid(message_on_source)
            message_on_target = torch.sigmoid(message_on_target)
        elif self.update_func == "relu":
            message_on_source = torch.nn.functional.relu(message_on_source)
            message_on_target = torch.nn.functional.relu(message_on_target)

        return message_on_source, message_on_target

    def attention(self, s_message, t_message):
        """Compute attention matrices A_s and A_t.

        Parameters
        ----------
        s_message : torch.Tensor, shape [source_cells, source_out_channels]
            Message tensor. This is the result of the matrix multiplication of the cochain matrix Xs with the weight
            matrix W_s.
        t_message : torch.Tensor, shape [target_cells, target_out_channels]
            Message tensor. This is the result of the matrix multiplication of the cochain matrix Xt with the weight
            matrix W_t.

        Returns
        -------
        A_s : torch.sparse, shape [target_cells, source_cells]. Represents the attention matrix A_s.
        A_t : torch.sparse, shape [source_cells, target_cells]. Represents the attention matrix A_t.
        """
        s_to_t = torch.cat(
            [s_message[self.source_indices], t_message[self.target_indices]], dim=1
        )

        t_to_s = torch.cat(
            [t_message[self.target_indices], s_message[self.source_indices]], dim=1
        )

        e = torch.sparse_coo_tensor(
            indices=torch.tensor(
                [self.target_indices.tolist(), self.source_indices.tolist()]
            ),
            values=F.leaky_relu(
                torch.matmul(s_to_t, self.att_weight),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(t_message.shape[0], s_message.shape[0]),
        )

        f = torch.sparse_coo_tensor(
            indices=torch.tensor(
                [self.source_indices.tolist(), self.target_indices.tolist()]
            ),
            values=F.leaky_relu(
                torch.matmul(
                    t_to_s,
                    torch.cat(
                        [
                            self.att_weight[self.source_out_channels :],
                            self.att_weight[: self.source_out_channels],
                        ]
                    ),
                ),
                negative_slope=self.negative_slope,
            ).squeeze(1),
            size=(s_message.shape[0], t_message.shape[0]),
        )

        if self.softmax:
            return torch.sparse.softmax(e, dim=1), torch.sparse.softmax(f, dim=1)

        return sparse_row_norm(e), sparse_row_norm(f)

    def forward(self, x_source, x_target, neighborhood):
        """Forward pass.

        The forward pass of the Higher Order Attention Block layer for non-squared matrices.
        x_source and x_target are the cochain matrices Xs and Xt used as input features for the layer. neighborhood is
        the neighborhood matrix A from source cells to target cells. Note that the neighborhood matrix shape
        should be [target_cells, source_cells] where target_cells and source_cells are the number of rows in x_target
        and x_source, respectively.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[source_cells, source_in_channels]
            Cochain matrix Xs used as input of the layer.
        x_target : torch.Tensor, shape=[target_cells, target_in_channels]
            Cochain matrix Xt used as input of the layer.
        neighborhood : torch.sparse, shape=[target_cells, source_cells]
            Neighborhood matrix used to compute the HBNS layer.

        Returns
        -------
        _ : Tensor, shape=[source_cells, source_out_channels]
            Output features of the layer for the source cells.
        _ : Tensor, shape=[target_cells, target_out_channels]
            Output features of the layer for the target cells.
        """
        s_message = torch.mm(x_source, self.w_s)  # [n_source_cells, d_t_out]
        t_message = torch.mm(x_target, self.w_t)  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = neighborhood.coalesce()
        neighborhood_t_to_s = neighborhood.t().coalesce()

        self.target_indices, self.source_indices = neighborhood_s_to_t.indices()

        s_to_t_attention, t_to_s_attention = self.attention(s_message, t_message)

        neighborhood_t_to_s_att = torch.sparse_coo_tensor(
            indices=neighborhood_t_to_s.indices(),
            values=t_to_s_attention.values() * neighborhood_t_to_s.values(),
            size=neighborhood_t_to_s.shape,
        )

        neighborhood_s_to_t_att = torch.sparse_coo_tensor(
            indices=neighborhood_s_to_t.indices(),
            values=s_to_t_attention.values() * neighborhood_s_to_t.values(),
            size=neighborhood_s_to_t.shape,
        )

        message_on_source = torch.mm(neighborhood_t_to_s_att, t_message)
        message_on_target = torch.mm(neighborhood_s_to_t_att, s_message)

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)
