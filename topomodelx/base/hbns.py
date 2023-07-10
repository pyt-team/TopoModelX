"""Higher Order Attention Block for non-squared neighborhoods (HBNS) for message passing module."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing


# TODO : This should be in a utils file. We keep it here for now to present the code to the challenge.
def sparse_row_norm(sparse_tensor):
    r"""Normalize a sparse tensor by row dividing each row by its sum.

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
    r"""Higher Order Attention Block for non-squared neighborhood matrices (HBNS). HBNS layers were introduced in [HAJIJ23]_, Definition 31 and 33.

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` as the :math:`d`-dimensional
    :math:`\mathbb{R}`-valued vector space of signals over the :math:`k`-th skeleton of :math:`\mathcal{X}`. Elements of this space are called :math:`k`-cochains of :math:`\mathcal{X}`.
    If :math:`d = 1`, we denote :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N: \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^t(\mathcal{X})` with :math:`s \neq t` be a non-squared neighborhood matrix from the space of signals over :math:`s`th-skeleton of :math:`\mathcal{X}` to the :math:`t`-skeleton of :math:`\mathcal{X}`.
    The higher order attention block induced by :math:`N` is a cochain map

    ..  math::
        \begin{align}
            HBNS_N: \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{in}}}) \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{t_{in}}}) \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{t_{out}}}) \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{s_{out}}}),
        \end{align}

    where :math:`d^{s_{in}}`, :math:`d^{t_{in}}`, :math:`d^{s_{out}}`, and :math:`d^{t_{out}}` are the input and output dimensions of the source and target cochains, respectively, also denoted as source_in_channels, target_in_channels, source_out_channels, and target_out_channels.

    The cochain map :math:`HBNS_N` is defined as

    ..  math::
        \begin{align}
            HBNS_N(X_s, X_t) = (Y_s, Y_t),
        \end{align}

    where the source and target output cochain matrices Y_s and Y_t are computed as

     ..  math::
        \begin{align}
            Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
            Y_t &= \phi((N \odot A_s) X_s W_s ).
        \end{align}

    Here, :math:`\odot` denotes the Hadamard product, namely the entry-wise product, and :math:`\phi` is a non-linear activation function.
    :math:`W_t` and :math:`W_s` are learnable weight matrices of shapes [target_in_channels, source_out_channels] and
    [source_in_channels, target_out_channels], respectively. Attention matrices are denoted as :math:`A_t` and :math:`A_s` and have the same dimensions as
    :math:`N^T` and :math:`N`, respectively. The entries :math:`(i, j)` of the attention matrices :math:`A_t` and :math:`A_s` are defined as

    ..  math::
        \begin{align}
            A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{columns(N)} e_{i,k}}, \\
            A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{columns(N^T)} f_{i,k}},
        \end{align}

    where,

    ..  math::
        \begin{align}
            e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
            f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[d_{s_{out}}:]||a[:d_{s_{out}}])),\\
        \end{align}

    where || denotes concatenation, :math:`a` is a learnable column vector of length
    :math:`d_{s_{out}} + d_{t_{out}}`. Given a vector :math:`v`, we denote by :math:`v[:c]` and :math:`v[c:]` to the projection onto the first :math:`c` elements and the last elements of :math:`v` starting from the :math:`(c+1)`-th element, respectively.
    S is the exponential function if softmax is used and the identity function otherwise.

    This is a sparse implementation of an HBNS layer.

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
        Whether to use softmax or sparse_row_norm in the computation of the attention matrix. Default is False.
    update_func : {None, 'sigmoid', 'relu'}, optional
        Activation function :math:`\phi` in the computation of the output of the layer. If None, :math:`\phi` is the identity function. Default is
        None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of :math:`W_p` and the attention vector :math:`a`. Default is 'xavier_uniform'.
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

    def get_device(self):
        """Get the device on which the layer's learnable parameters are stored."""
        return self.w_s.device

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
        """Update embeddings on each cell with an activation function, either sigmoid or ReLU.

        Parameters
        ----------
        message_on_source : torch.Tensor, shape=[source_cells, source_out_channels]
            Output features of the layer for the source before the update function.
        message_on_target : torch.Tensor, shape=[target_cells, target_out_channels]
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
        elif self.update_func == "tanh":
            message_on_source = torch.nn.functional.tanh(message_on_source)
            message_on_target = torch.nn.functional.tanh(message_on_target)

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
            device=self.get_device(),
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
            device=self.get_device(),
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
            device=self.get_device(),
        )

        neighborhood_s_to_t_att = torch.sparse_coo_tensor(
            indices=neighborhood_s_to_t.indices(),
            values=s_to_t_attention.values() * neighborhood_s_to_t.values(),
            size=neighborhood_s_to_t.shape,
            device=self.get_device(),
        )

        message_on_source = torch.mm(neighborhood_t_to_s_att, t_message)
        message_on_target = torch.mm(neighborhood_s_to_t_att, s_message)

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)
