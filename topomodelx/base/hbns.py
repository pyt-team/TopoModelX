"""Higher Order Attention Block for non-squared neighborhood matrices (HBNS)."""


import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.message_passing import MessagePassing

from ..utils.srn import sparse_row_norm


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

    where the source and target output cochain matrices :math:`Y_s` and :math:`Y_t` are computed as

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

    This HBNS class just contains the sparse implementation of the block.

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
        source_in_channels: int,
        source_out_channels: int,
        target_in_channels: int,
        target_out_channels: int,
        negative_slope: float = 0.2,
        softmax: bool = False,
        update_func: str = None,
        initialization: str = "xavier_uniform",
    ) -> None:
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

    def get_device(self) -> torch.device:
        """Get the device on which the layer's learnable parameters are stored."""
        return self.w_s.device

    def reset_parameters(self, gain=1.414) -> None:
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

    def update(
        self, message_on_source: torch.Tensor, message_on_target: torch.Tensor
    ) -> tuple[torch.Tensor]:
        r"""Update signal features on each cell with an activation function, either sigmoid, ReLU or tanh.

        Parameters
        ----------
        message_on_source : torch.Tensor, shape=[source_cells, source_out_channels]
            Source output signal features before the activation function :math:`\phi`.
        message_on_target : torch.Tensor, shape=[target_cells, target_out_channels]
            Target output signal features before the activation function :math:`\phi`.

        Returns
        -------
        _ phi(Y_s) : torch.Tensor, shape=[source_cells, source_out_channels]
            Source output signal features after the activation function :math:`\phi`.
        _ phi(Y_t) : torch.Tensor, shape=[target_cells, target_out_channels]
            Target output signal features after the activation function :math:`\phi`.
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

    def attention(
        self, s_message: torch.Tensor, t_message: torch.Tensor
    ) -> tuple[torch.Tensor]:
        r"""Compute attention matrices :math:`A_s` and :math:`A_t`.

        ..  math::
            \begin{align}
                A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{columns(N)} e_{i,k}}, \\
                A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{columns(N^T)} f_{i,k}},
            \end{align}

        where,

        ..  math::
            \begin{align}
                e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
                f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[d_{s_{out}}:]||a[:d_{s_{out}}])).
            \end{align}


        Parameters
        ----------
        s_message : torch.Tensor, shape [n_source_cells, target_out_channels]
            Source message tensor. This is the result of the matrix multiplication of the cochain matrix :math:`X_s` with the weight
            matrix :math:`W_s`.
        t_message : torch.Tensor, shape [n_target_cells, source_out_channels]
            Target message tensor. This is the result of the matrix multiplication of the cochain matrix :math:`X_t` with the weight
            matrix :math:`W_t`.

        Returns
        -------
        A_s : torch.sparse, shape=[target_cells, source_cells].
        A_t : torch.sparse, shape=[source_cells, target_cells].
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

    def forward(
        self, x_source: torch.Tensor, x_target: torch.Tensor, neighborhood: torch.Tensor
    ) -> tuple[torch.Tensor]:
        r"""Forward pass of the Higher Order Attention Block for non-squared matrices.

        The forward pass computes:

        ..  math::
            \begin{align}
                HBNS_N(X_s, X_t) = (Y_s, Y_t),
            \end{align}

        where the source and target outputs :math:`Y_s` and :math:`Y_t` are computed as

        ..  math::
            \begin{align}
                Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
                Y_t &= \phi((N \odot A_s) X_s W_s ).
            \end{align}

        Parameters
        ----------
        x_source : torch.Tensor, shape=[source_cells, source_in_channels]
            Cochain matrix representation :math:`X_s` containing the signal features over the source cells.
        x_target : torch.Tensor, shape=[target_cells, target_in_channels]
            Cochain matrix :math:`X_t` containing the signal features over the target cells.
        neighborhood : torch.sparse, shape=[target_cells, source_cells]
            Neighborhood matrix :math:`N` inducing the HBNS block.

        Returns
        -------
        _ :math:`Y_s` : torch.Tensor, shape=[source_cells, source_out_channels]
            Output features of the layer for the source cells.
        _ :math:`Y_t` : torch.Tensor, shape=[target_cells, target_out_channels]
            Output features of the layer for the target cells.
        """
        s_message = torch.mm(x_source, self.w_s)  # [n_source_cells, d_t_out]
        t_message = torch.mm(x_target, self.w_t)  # [n_target_cells, d_s_out]

        neighborhood_s_to_t = (
            neighborhood.coalesce()
        )  # [n_target_cells, n_source_cells]
        neighborhood_t_to_s = (
            neighborhood.t().coalesce()
        )  # [n_source_cells, n_target_cells]

        self.target_indices, self.source_indices = neighborhood_s_to_t.indices()

        s_to_t_attention, t_to_s_attention = self.attention(s_message, t_message)

        neighborhood_s_to_t_att = torch.sparse_coo_tensor(
            indices=neighborhood_s_to_t.indices(),
            values=s_to_t_attention.values() * neighborhood_s_to_t.values(),
            size=neighborhood_s_to_t.shape,
            device=self.get_device(),
        )

        neighborhood_t_to_s_att = torch.sparse_coo_tensor(
            indices=neighborhood_t_to_s.indices(),
            values=t_to_s_attention.values() * neighborhood_t_to_s.values(),
            size=neighborhood_t_to_s.shape,
            device=self.get_device(),
        )

        message_on_source = torch.mm(neighborhood_t_to_s_att, t_message)
        message_on_target = torch.mm(neighborhood_s_to_t_att, s_message)

        if self.update_func is None:
            return message_on_source, message_on_target

        return self.update(message_on_source, message_on_target)
