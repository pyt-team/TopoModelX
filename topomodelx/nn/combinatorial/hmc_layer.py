"""Higher-Order Attentional NN Layer for Mesh Classification."""

import torch
import torch.nn.functional as F
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation


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


class HBNS(torch.nn.Module):
    r"""Higher Order Attention Block for non-squared neighborhood matrices.

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote by
    :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` the :math:`\mathbb{
    R}`-valued vector space of :math:`d`-dimensional signals over
    :math:`\Sigma^k`, the :math:`k`-th skeleton of :math:`\mathcal{X}`
    subject to a certain total order. Elements of this space are called
    :math:`k`-cochains of :math:`\mathcal{X}`. If :math:`d = 1`, we denote
    it by :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N: \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^t(
    \mathcal{X})` with :math:`s \neq t` be a non-squared neighborhood matrix
    from the :math:`s` th-skeleton of :math:`\mathcal{
    X}` to its :math:`t` th-skeleton. The higher order
    attention block induced by :math:`N` is a cochain map

    ..  math::
        \begin{align}
            \text{HBNS}_N: \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{in}}})
            \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{t_{in}}})
            \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{t_{out}}})
            \times \mathcal{C}^t(\mathcal{X},\mathbb{R}^{d^{s_{out}}}),
        \end{align}

    where :math:`d^{s_{in}}`, :math:`d^{t_{in}}`, :math:`d^{s_{out}}`,
    and :math:`d^{t_{out}}` are the input and output dimensions of the
    source and target cochains, also denoted as source_in_channels,
    target_in_channels, source_out_channels, and target_out_channels.

    The cochain map :math:`\text{HBNS}_N` is defined as

    ..  math::
        \begin{align}
            \text{HBNS}_N(X_s, X_t) = (Y_s, Y_t),
        \end{align}

    where the source and target output cochain matrices :math:`Y_s` and
    :math:`Y_t` are computed as

     ..  math::
        \begin{align}
            Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
            Y_t &= \phi((N \odot A_s) X_s W_s ).
        \end{align}

    Here, :math:`\odot` denotes the Hadamard product, namely the entry-wise
    product, and :math:`\phi` is a non-linear activation function.
    :math:`W_t` and :math:`W_s` are learnable weight matrices of shapes
    [target_in_channels, source_out_channels] and [source_in_channels,
    target_out_channels], respectively. Attention matrices are denoted as
    :math:`A_t` and :math:`A_s` and have the same dimensions as :math:`N^T`
    and :math:`N`, respectively. The entries :math:`(i, j)` of the attention
    matrices :math:`A_t` and :math:`A_s` are defined as

    ..  math::
        \begin{align}
            A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{\#\text{columns}(N)} e_{i,
            k}}, \\
            A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{\#\text{columns}(N^T)} f_{i,
            k}},
        \end{align}

    where,

    ..  math::
        \begin{align}
            e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
            f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[d_{s_{
            out}}:]||a[:d_{s_{out}}])).\\
        \end{align}

    Here, || denotes concatenation and :math:`a` denotes the learnable column
    attention vector of length :math:`d_{s_{out}} + d_{t_{out}}`.
    Given a vector :math:`v`, we denote by :math:`v[:c]` and :math:`v[c:]`
    to the projection onto the first :math:`c` elements and the last
    elements of :math:`v` starting from the :math:`(c+1)`-th element,
    respectively. :math:`S` is the exponential function if softmax is used
    and the identity function otherwise.

    This HBNS class just contains the sparse implementation of the block.

    Notes
    -----
    HBNS layers were introduced in [H23]_, Definition 31 and 33.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

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
        Whether to use softmax or sparse_row_norm in the computation of the
        attention matrix. Default is False.
    update_func : {None, 'sigmoid', 'relu'}, optional
        Activation function :math:`\phi` in the computation of the output of
        the layer. If None, :math:`\phi` is the identity function. Default is
        None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of :math:`W_p` and the attention
        vector :math:`a`. Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        target_in_channels: int,
        target_out_channels: int,
        negative_slope: float = 0.2,
        softmax: bool = False,
        update_func: str | None = None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.initialization = initialization

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
        """Get device on which the layer's learnable parameters are stored."""
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
                "Initialization method not recognized."
                "Should be either xavier_uniform or xavier_normal."
            )

    def update(
        self, message_on_source: torch.Tensor, message_on_target: torch.Tensor
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""Update signal features on each cell with an activation function.

        The implemented activation functions are sigmoid, ReLU and tanh.

        Parameters
        ----------
        message_on_source : torch.Tensor, shape=[source_cells,
        source_out_channels]
            Source output signal features before the activation function
            :math:`\phi`.
        message_on_target : torch.Tensor, shape=[target_cells,
        target_out_channels]
            Target output signal features before the activation function
            :math:`\phi`.

        Returns
        -------
        phi(Y_s) : torch.Tensor, shape=[source_cells, source_out_channels]
            Source output signal features after the activation function
            :math:`\phi`.
        phi(Y_t) : torch.Tensor, shape=[target_cells, target_out_channels]
            Target output signal features after the activation function
            :math:`\phi`.
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
    ) -> tuple[torch.sparse.FloatTensor, torch.sparse.FloatTensor]:
        r"""Compute attention matrices :math:`A_s` and :math:`A_t`.

        ..  math::
            \begin{align}
                A_s(i,j) &= \frac{e_{i,j}}{\sum_{k=1}^{\#\text{columns}(N)}
                e_{i,k}}, \\
                A_t(i,j) &= \frac{f_{i,j}}{\sum_{k=1}^{\#\text{columns}(N^T)}
                f_{i,k}},
            \end{align}

        where,

        ..  math::
            \begin{align}
                e_{i,j} &= S(\text{LeakyReLU}([(X_s)_jW_s||(X_t)_iW_t]a)),\\
                f_{i,j} &= S(\text{LeakyReLU}([(X_t)_jW_t||(X_s)_iW_s][a[
                d_{s_{out}}:]||a[:d_{s_{out}}])).
            \end{align}

        Parameters
        ----------
        s_message : torch.Tensor, shape [n_source_cells, target_out_channels]
            Source message tensor. This is the result of the matrix
            multiplication of the cochain matrix :math:`X_s` with the weight
            matrix :math:`W_s`.
        t_message : torch.Tensor, shape [n_target_cells, source_out_channels]
            Target message tensor. This is the result of the matrix
            multiplication of the cochain matrix :math:`X_t` with the weight
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
    ) -> tuple[torch.FloatTensor, torch.FloatTensor]:
        r"""Compute forward pass.

        The forward pass of the Higher Order Attention Block for non-squared
        matrices (HBNS) is defined as

        ..  math::
            \begin{align}
                \text{HBNS}_N(X_s, X_t) = (Y_s, Y_t),
            \end{align}

        where the source and target outputs :math:`Y_s` and :math:`Y_t` are
        computed as

        ..  math::
            \begin{align}
                Y_s &= \phi((N^T \odot A_t) X_t W_t), \\
                Y_t &= \phi((N \odot A_s) X_s W_s ).
            \end{align}

        Parameters
        ----------
        x_source : torch.Tensor, shape=[source_cells, source_in_channels]
            Cochain matrix representation :math:`X_s` containing the signal
            features over the source cells.
        x_target : torch.Tensor, shape=[target_cells, target_in_channels]
            Cochain matrix :math:`X_t` containing the signal features over
            the target cells.
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


class HBS(torch.nn.Module):
    r"""Higher Order Attention Block layer for squared neighborhoods (HBS).

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote by
    :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` the :math:`\mathbb{
    R}`-valued vector space of :math:`d`-dimensional signals over
    :math:`\Sigma^k`, the :math:`k`-th skeleton of :math:`\mathcal{X}`
    subject to a certain total order. Elements of this space are called
    :math:`k`-cochains of :math:`\mathcal{X}`. If :math:`d = 1`, we denote
    it by :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N\colon \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^s(
    \mathcal{X})` be a cochain map endomorphism of the space of signals over
    :math:`\Sigma^s` of :math:`\mathcal{X}`. The matrix representation of
    :math:`N` has shape :math:`n_{cells} \times n_{cells}`, where :math:`n_{
    cells}` denotes the cardinality of :math:`\Sigma^s`.

    The higher order attention block induced by :math:`N` is the cochain map

    ..  math::
        \begin{align}
            \text{HBS}_N\colon \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{
            in}}}) \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{
            out}}}),
        \end{align}

    where :math:`d^{s_{in}}` and :math:`d^{s_{out}}` are the input and
    output dimensions of the HBS block, also denoted as
    source_in_channels and source_out_channels, respectively.

    :math:`\text{HBS}_N` is defined by

    ..  math::
        \phi(\sum_{p=1}^{\text{m_hop}}(N^p \odot A_p) X W_p )

    where :math:`X` is the cochain matrix representation of shape [n_cells,
    source_in_channels] under the canonical basis of :math:`\mathcal{C}^s(
    \mathcal{X},\mathbb{R}^{d^{s_{in}}})`, induced by the total order of
    :math:`\Sigma^s`, that contains the input features for each cell. The
    :math:`\odot` symbol denotes the Hadamard product, namely the entry-wise
    product, and :math:`\phi` is a non-linear activation function.
    :math:`W_p` is a learnable weight matrix of shape [source_in_channels,
    source_out_channels] for each :math:`p`, and :math:`A_p` is an attention
    matrix with the same dimensionality as the input neighborhood matrix
    :math:`N`, i.e., [n_cells, n_cells]. The indices :math:`(i,j)` of the
    attention matrix :math:`A_p` are computed as

    ..  math::
        A_p(i,j) = \frac{e_{i,j}^p}{\sum_{k=1}^{\#\text{columns}(N)} e_{i,k}^p}

    where

    ..  math::
        e_{i,j}^p = S(\text{LeakyReLU}([X_iW_p||X_jW_p]a_p))

    and where || denotes concatenation, :math:`a_p` is a learnable column
    vector of length :math:`2\cdot` source_out_channels, and :math:`S` is the
    exponential function if softmax is used and the identity function
    otherwise.

    This HBS class just contains the sparse implementation of the block.

    Notes
    -----
    HBS layers were introduced in [H23]_, Definitions 31 and 32.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

    Parameters
    ----------
    source_in_channels : int
        Number of input features for the source cells.
    source_out_channels : int
        Number of output features for the source cells.
    negative_slope : float
        Negative slope of the LeakyReLU activation function.
    softmax : bool, optional
        Whether to use softmax in the computation of the attention matrix.
        Default is False.
    m_hop : int, optional
        Maximum number of hops to consider in the computation of the layer
        function. Default is 1.
    update_func : {None, 'sigmoid', 'relu', 'tanh'}, optional
        Activation function :math:`phi` in the computation of the output of
        the layer.
        If None, :math:`phi` is the identity function. Default is None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of W_p and :math:`a_p`.
        Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        source_in_channels: int,
        source_out_channels: int,
        negative_slope: float = 0.2,
        softmax: bool = False,
        m_hop: int = 1,
        update_func: str = None,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()

        self.initialization = initialization

        self.source_in_channels = source_in_channels
        self.source_out_channels = source_out_channels

        self.m_hop = m_hop
        self.update_func = update_func

        self.weight = torch.nn.ParameterList(
            [
                Parameter(
                    torch.Tensor(self.source_in_channels, self.source_out_channels)
                )
                for _ in range(self.m_hop)
            ]
        )

        self.att_weight = torch.nn.ParameterList(
            [
                Parameter(torch.Tensor(2 * self.source_out_channels, 1))
                for _ in range(self.m_hop)
            ]
        )
        self.negative_slope = negative_slope
        self.softmax = softmax

        self.reset_parameters()

    def get_device(self) -> torch.device:
        """Get device on which the layer's learnable parameters are stored."""
        return self.weight[0].device

    def reset_parameters(self, gain: float = 1.414) -> None:
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization. Default is 1.414.
        """

        def reset_p_hop_parameters(weight, att_weight):
            if self.initialization == "xavier_uniform":
                torch.nn.init.xavier_uniform_(weight, gain=gain)
                torch.nn.init.xavier_uniform_(att_weight.view(-1, 1), gain=gain)

            elif self.initialization == "xavier_normal":
                torch.nn.init.xavier_normal_(weight, gain=gain)
                torch.nn.init.xavier_normal_(att_weight.view(-1, 1), gain=gain)
            else:
                raise RuntimeError(
                    "Initialization method not recognized. "
                    "Should be either xavier_uniform or xavier_normal."
                )

        for w, a in zip(self.weight, self.att_weight):
            reset_p_hop_parameters(w, a)

    def update(self, message: torch.Tensor) -> torch.Tensor:
        r"""Update signal features on each cell with an activation function.

        Implemented activation functions are sigmoid, ReLU and tanh.

        Parameters
        ----------
        message : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features before the activation function :math:`\phi`.

        Returns
        -------
        _ : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features after the activation function :math:`\phi`.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(message)
        if self.update_func == "relu":
            return torch.nn.functional.relu(message)
        if self.update_func == "tanh":
            return torch.nn.functional.tanh(message)
        else:
            raise RuntimeError(
                "Update activation function not "
                "recognized. Should be either sigmoid, "
                "relu or tanh."
            )

    def attention(
        self, message: torch.Tensor, A_p: torch.Tensor, a_p: torch.Tensor
    ) -> torch.sparse.FloatTensor:
        """Compute the attention matrix.

        Parameters
        ----------
        message : torch.Tensor, shape=[n_messages, source_out_channels]
            Message tensor. This is the result of the matrix multiplication
            of the cochain matrix :math:`X` with the learnable weights
            matrix :math:`W_p`.
        A_p : torch.sparse, shape=[n_cells, n_cells]
            Neighborhood matrix to the power p. Indicates how many paths of
            lenght p exist from cell :math:`i` to cell :math:`j`.
        a_p : torch.Tensor, shape=[2*source_out_channels, 1]
            Learnable attention weight vector.

        Returns
        -------
        att_p : torch.sparse, shape=[n_messages, n_messages].
            Represents the attention matrix :math:`A_p`.
        """
        n_messages = message.shape[0]
        source_index_i, source_index_j = A_p.coalesce().indices()
        s_to_s = torch.cat([message[source_index_i], message[source_index_j]], dim=1)
        e_p = torch.sparse_coo_tensor(
            indices=torch.tensor([source_index_i.tolist(), source_index_j.tolist()]),
            values=F.leaky_relu(
                torch.matmul(s_to_s, a_p), negative_slope=self.negative_slope
            ).squeeze(1),
            size=(n_messages, n_messages),
            device=self.get_device(),
        )
        att_p = (
            torch.sparse.softmax(e_p, dim=1) if self.softmax else sparse_row_norm(e_p)
        )

        return att_p

    def forward(
        self, x_source: torch.Tensor, neighborhood: torch.Tensor
    ) -> torch.FloatTensor:
        r"""Compute forward pass.

        The forward pass of the Higher Order Attention Block for squared
        neighborhood matrices is defined as:

        ..  math::
            \text{HBS}_N(X) = \phi(\sum_{p=1}^{\text{m_hop}}(N^p \odot A_p) X
            W_p ).

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_cells, source_in_channels]
            Cochain matrix representation :math:`X` whose rows correspond to
            the signal features over each cell following the order of the cells
            in :math:`\Sigma^s`.
        neighborhood : torch.sparse, shape=[n_cells, n_cells]
            Neighborhood matrix :math:`N`.

        Returns
        -------
        _ : Tensor, shape=[n_cells, source_out_channels]
            Output features of the layer.
        """
        message = [
            torch.mm(x_source, w) for w in self.weight
        ]  # [m-hop, n_source_cells, d_t_out]

        # Create a torch.eye with the device of x_source
        A_p = torch.eye(x_source.shape[0], device=self.get_device()).to_sparse_coo()

        m_hop_matrices = []

        # Generate the list of neighborhood matrices :math:`A, \dots, A^m`
        for _ in range(self.m_hop):
            A_p = torch.sparse.mm(A_p, neighborhood)
            m_hop_matrices.append(A_p)

        att = [
            self.attention(m_p, A_p, a_p)
            for m_p, A_p, a_p in zip(message, m_hop_matrices, self.att_weight)
        ]

        def sparse_hadamard(A_p, att_p):
            return torch.sparse_coo_tensor(
                indices=A_p.indices(),
                values=att_p.values() * A_p.values(),
                size=A_p.shape,
                device=self.get_device(),
            )

        att_m_hop_matrices = [
            sparse_hadamard(A_p, att_p) for A_p, att_p in zip(m_hop_matrices, att)
        ]

        message = [torch.mm(n_p, m_p) for n_p, m_p in zip(att_m_hop_matrices, message)]
        result = torch.zeros_like(message[0])

        for m_p in message:
            result += m_p

        if self.update_func is None:
            return result

        return self.update(result)


class HMCLayer(torch.nn.Module):
    r"""Higher-Order Attentional NN Layer for Mesh Classification.

    Implementation of the Message Passing Layer for the Combinatorial Complex
    Attentional Neural Network for Mesh Classification, introduced in
    [H23]_ and graphically illustrated in Figure 35(b) of [H23]_.

    The layer is composed of two stacked levels of attentional message passing
    steps, which in both cases, update the signal features over the cells of the
    zeroth, first and second skeleton of the combinatorial complex.

    The attentional message passing steps performed in each level are:

    Level 1. 0-dimensional cells (vertices) receive messages from 0-dimensional
    cells (vertices) and from 1-dimensional cells (edges). In the first
    case, adjacency matrices are used. In the second case, the incidence
    matrix from dimension 1 to dimension 0 is used. 1-dimensional cells
    (edges) receive messages from 1-dimensional cells (edges) and from
    2-dimensional cells (faces). In both cases, incidence matrices are
    used. 2-dimensional cells (faces) receive messages only from
    1-dimensional cells (edges). In this case, the incidence matrix
    from dimension 2 to dimension 1 is used.

    Level 2. 0-dimensional cells (vertices) receive messages from 0-dimensional
    cells (vertices) using their adjacency matrix.
    1-dimensional cells (edges) receive messages from 0-dimensional
    cells (vertices) and from 1-dimensional cells (edges) using
    incidence and adjacency matrices, respectively. 2-dimensional cells
    (faces) receive messages from 1-dimensional cells (edges) and from
    2-dimensional cells (faces) using incidence and coadjacency
    matrices, respectively.

    Notes
    -----
    This is the architecture proposed for mesh classification. Meshes are
    assumed to be represented as combinatorial complexes.

    References
    ----------
    .. [H23] Hajij, Zamzmi, Papamarkou, Miolane, Guzmán-Sáenz, Ramamurthy, Birdal, Dey,
        Mukherjee, Samaga, Livesay, Walters, Rosen, Schaub. Topological Deep Learning: Going Beyond Graph Data.
        (2023) https://arxiv.org/abs/2206.00606.

    .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
        Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
        (2023) https://arxiv.org/abs/2304.10031.

    Parameters
    ----------
    in_channels : list of int
        Dimension of input features on vertices (0-cells), edges (
        1-cells) and faces (2-cells). The length of the list
        must be 3.
    intermediate_channels : list of int
        Dimension of intermediate features on vertices (0-cells),
        edges (1-cells) and faces (2-cells). The length of the
        list must be 3. The intermediate features are the ones computed
        after the first step of message passing.
    out_channels : list of int
        Dimension of output features on vertices (0-cells), edges (
        1-cells) and faces (2-cells). The length of the list must be 3.
        The output features are the ones computed after the second step
        of message passing.
    negative_slope : float
        Negative slope of LeakyReLU used to compute the attention
        coefficients.
    softmax_attention : bool, optional
        Whether to use softmax attention. If True, the attention
        coefficients are normalized by rows using softmax over all the
        columns that are not zero in the associated neighborhood
        matrix. If False, the normalization is done by dividing by the
        sum of the values of the coefficients in its row whose columns
        are not zero in the associated neighborhood matrix. Default is
        False.
    update_func_attention : string, optional
        Activation function used in the attention block. If None,
        no activation function is applied. Default is None.
    update_func_aggregation : string, optional
        Function used to aggregate the messages computed in each
        attention block. If None, the messages are aggregated by summing
        them. Default is None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of the attention layers.
        Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        in_channels: list[int],
        intermediate_channels: list[int],
        out_channels: list[int],
        negative_slope: float,
        softmax_attention=False,
        update_func_attention=None,
        update_func_aggregation=None,
        initialization="xavier_uniform",
    ):
        super(HMCLayer, self).__init__()
        super().__init__()

        assert (
            len(in_channels) == 3
            and len(intermediate_channels) == 3
            and len(out_channels) == 3
        )

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        (
            intermediate_channels_0,
            intermediate_channels_1,
            intermediate_channels_2,
        ) = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.hbs_0_level1 = HBS(
            source_in_channels=in_channels_0,
            source_out_channels=intermediate_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_0_1_level1 = HBNS(
            source_in_channels=in_channels_1,
            source_out_channels=intermediate_channels_1,
            target_in_channels=in_channels_0,
            target_out_channels=intermediate_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_1_2_level1 = HBNS(
            source_in_channels=in_channels_2,
            source_out_channels=intermediate_channels_2,
            target_in_channels=in_channels_1,
            target_out_channels=intermediate_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_0_level2 = HBS(
            source_in_channels=intermediate_channels_0,
            source_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_0_1_level2 = HBNS(
            source_in_channels=intermediate_channels_1,
            source_out_channels=out_channels_1,
            target_in_channels=intermediate_channels_0,
            target_out_channels=out_channels_0,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_1_level2 = HBS(
            source_in_channels=intermediate_channels_1,
            source_out_channels=out_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbns_1_2_level2 = HBNS(
            source_in_channels=intermediate_channels_2,
            source_out_channels=out_channels_2,
            target_in_channels=intermediate_channels_1,
            target_out_channels=out_channels_1,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.hbs_2_level2 = HBS(
            source_in_channels=intermediate_channels_2,
            source_out_channels=out_channels_2,
            negative_slope=negative_slope,
            softmax=softmax_attention,
            update_func=update_func_attention,
            initialization=initialization,
        )

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        adjacency_0,
        adjacency_1,
        coadjacency_2,
        incidence_1,
        incidence_2,
    ):
        r"""Forward pass.

        The forward pass of the Combinatorial Complex Attention Neural
        Network for Mesh Classification proposed in [H23]_, Figure 35(
        b). The input features are transformed in two consecutive stacked
        levels of attentional message passing steps, which update the
        signal features over the cells of the zeroth, first and second
        skeletons of the combinatorial complex.

        Following the notations of [PSHM23]_, the steps for each level can be
        summarized as follows:

        1.First level:

        ..  math::
            \begin{align}
                &🟥\quad m^{0\rightarrow 0}_{y\rightarrow x} = \left((A_{
                \uparrow,0})_{xy} \cdot \text{att}_{xy}^{0\rightarrow
                0}\right) h_y^{t,(0)} \Theta^t_{0\rightarrow 0}\\
                &🟥\quad m^{0\rightarrow 1}_{y\rightarrow x} = \left((B_{
                1}^T)_{xy} \cdot \text{att}_{xy}^{0\rightarrow 1}\right)
                h_y^{t,(0)} \Theta^t_{0\rightarrow 1}\\
                &🟥\quad m^{1\rightarrow 0}_{y\rightarrow x} = \left((B_{
                1})_{xy} \cdot \text{att}_{xy}^{1\rightarrow 0}\right) h_y^{t,
                (1)} \Theta^t_{1\rightarrow 0}\\
                &🟥\quad m^{1\rightarrow 2}_{y\rightarrow x} = \left((B_{
                2}^T)_{xy} \cdot \text{att}_{xy}^{1\rightarrow 2}\right)
                h_y^{t,(1)} \Theta^t_{1\rightarrow 2}\\
                &🟥\quad m^{2\rightarrow 1}_{y\rightarrow x} = \left((B_{2})_{
                xy} \cdot \text{att}_{xy}^{2\rightarrow 1}\right) h_y^{t,
                (2)} \Theta^t_{2\rightarrow 1}\\
                &🟧\quad m^{0\rightarrow 0}_{x}=\phi_u\left(\sum_{y\in A_{
                \uparrow,0}(x)} m^{0\rightarrow 0}_{y\rightarrow x}\right)\\
                &🟧\quad m^{0\rightarrow 1}_{x}=\phi_u\left(\sum_{y\in B_{
                1}^T(x)} m^{0\rightarrow 1}_{y\rightarrow x}\right)\\
                &🟧\quad m^{1\rightarrow 0}_{x}=\phi_u\left(\sum_{y\in B_{
                1}(x)} m^{1\rightarrow 0}_{y\rightarrow x}\right)\\
                &🟧\quad m^{1\rightarrow 2}_{x}=\phi_u\left(\sum_{y\in B_{
                2}^T(x)} m^{1\rightarrow 2}_{y\rightarrow x}\right)\\
                &🟧\quad m^{2\rightarrow 1}_{x}=\phi_u\left(\sum_{y\in B_{
                2}(x)} m^{2\rightarrow 1}_{y\rightarrow x}\right)\\
                &🟩\quad m_x^{(0)}=\phi_a\left(m^{0\rightarrow 0}_{x}+m^{
                1\rightarrow 0}_{x}\right)\\
                &🟩\quad m_x^{(1)}=\phi_a\left(m^{0\rightarrow 1}_{x}+m^{
                2\rightarrow 1}_{x}\right)\\
                &🟩\quad m_x^{(2)}=\phi_a\left(m^{1\rightarrow 2}_{x}\right)\\
                &🟦\quad i_x^{t,(0)} = m_x^{(0)}\\
                &🟦\quad i_x^{t,(1)} = m_x^{(1)}\\
                &🟦\quad i_x^{t,(2)} = m_x^{(2)}
             \end{align}

        where :math:`i_x^{t,(\cdot)}` represents intermediate feature vectors.

        2. Second level:

        ..  math::
            \begin{align}
                &🟥\quad m^{0\rightarrow 0}_{y\rightarrow x} = \left((A_{
                \uparrow,0})_{xy} \cdot \text{att}_{xy}^{0\rightarrow 0}\right)
                i_y^{t,(0)} \Theta^t_{0\rightarrow 0}\\
                &🟥\quad m^{1\rightarrow 1}_{y\rightarrow x} = \left((A_{
                \uparrow,1})_{xy} \cdot \text{att}_{xy}^{1\rightarrow 1}\right)
                i_y^{t,(1)} \Theta^t_{1\rightarrow 1}\\
                &🟥\quad m^{2\rightarrow 2}_{y\rightarrow x} = \left((A_{
                \downarrow, 2})_{xy} \cdot \text{att}_{xy}^{2\rightarrow
                2}\right) i_y^{t,(2)} \Theta^t_{2\rightarrow 2}\\
                &🟥\quad m^{0\rightarrow 1}_{y\rightarrow x} = \left((B_{
                1}^T)_{xy} \cdot \text{att}_{xy}^{0\rightarrow 1}\right)
                i_y^{t,(0)} \Theta^t_{0\rightarrow 1}\\
                &🟥\quad m^{1\rightarrow 2}_{y\rightarrow x} = \left((B_{
                2}^T)_{xy} \cdot \text{att}_{xy}^{1\rightarrow 2}\right)
                i_y^{t,(1)} \Theta^t_{1\rightarrow 2}\\
                &🟧\quad m^{0\rightarrow 0}_{x} = \phi_u\left(\sum_{y\in A_{
                \uparrow, 0}(x)} m^{0\rightarrow 0}_{y\rightarrow x}\right)\\
                &🟧\quad m^{1\rightarrow 1}_{x} = \phi_u\left(\sum_{y\in A_{
                \uparrow, 1}(x)} m^{1\rightarrow 1}_{y\rightarrow x}\right)\\
                &🟧\quad m^{2\rightarrow 2}_{x} = \phi_u\left(\sum_{y\in A_{
                \downarrow, 2}(x)} m^{2\rightarrow 2}_{y\rightarrow x}\right)\\
                &🟧\quad m^{0\rightarrow 1}_{x} = \phi_u\left(\sum_{y\in B_{
                1}^T(x)} m^{0\rightarrow 1}_{y\rightarrow x}\right)\\
                &🟧\quad m^{1\rightarrow 2}_{x} = \phi_u\left(\sum_{y\in B_{
                2}^T(x)} m^{1\rightarrow 2}_{y\rightarrow x}\right)\\
                &🟩\quad m_x^{(0)} = \phi_a\left(m^{0\rightarrow 0}_{x}+m^{
                1\rightarrow 0}_{x}\right)\\
                &🟩\quad m_x^{(1)} = \phi_a\left(m^{1\rightarrow 1}_{x} + m^{
                0\rightarrow 1}_{x}\right)\\
                &🟩\quad m_x^{(2)} = \phi_a\left(m^{1\rightarrow 2}_{x} + m^{
                2\rightarrow 2}_{x}\right)\\
                &🟦\quad h_x^{t+1,(0)} = m_x^{(0)}\\
                &🟦\quad h_x^{t+1,(1)} = m_x^{(1)}\\
                &🟦\quad h_x^{t+1,(2)} = m_x^{(2)}
            \end{align}

        In both message passing levels, :math:`\phi_u` and :math:`\phi_a`
        represent common activation functions within and between neighborhood
        aggregations. Both are passed to the constructor of the class as
        arguments update_func_attention and update_func_aggregation,
        respectively.

        References
        ----------
        .. [H23] Mustafa Hajij et al. Topological Deep Learning: Going
        Beyond Graph Data.
            arXiv:2206.00606.
            https://arxiv.org/pdf/2206.00606v3.pdf

        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on
            Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_0 : torch.Tensor, shape=[n_0_cells, in_channels[0]]
            Input features on the 0-cells (vertices) of the combinatorial
            complex.
        x_1 : torch.Tensor, shape=[n_1_cells, in_channels[1]]
            Input features on the 1-cells (edges) of the combinatorial complex.
        x_2 : torch.Tensor, shape=[n_2_cells, in_channels[2]]
        Input features on the 2-cells (faces) of the combinatorial complex.
        adjacency_0 : torch.sparse
            shape=[n_0_cells, n_0_cells]
            Neighborhood matrix mapping 0-cells to 0-cells (A_0_up).
        adjacency_1 : torch.sparse
            shape=[n_1_cells, n_1_cells]
            Neighborhood matrix mapping nodes to nodes (A_1_up).
        coadjacency_2 : torch.sparse
            shape=[n_2_cells, n_2_cells]
            Neighborhood matrix mapping nodes to nodes (A_2_down).
        incidence_1 : torch.sparse
            shape=[n_0_cells, n_1_cells]
            Neighborhood matrix mapping 1-cells to 0-cells (B_1).
        incidence_2 : torch.sparse
        shape=[n_1_cells, n_2_cells]
        Neighborhood matrix mapping 2-cells to 1-cells (B_2).

        Returns
        -------
        _ : torch.Tensor, shape=[1, num_classes]
            Output prediction on the entire cell complex.
        """
        # Computing messages from Higher Order Attention Blocks Level 1
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_0_to_1, x_1_to_0 = self.hbns_0_1_level1(x_1, x_0, incidence_1)
        x_1_to_2, x_2_to_1 = self.hbns_1_2_level1(x_2, x_1, incidence_2)

        x_0_level1 = self.aggr([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr([x_0_to_1, x_2_to_1])
        x_2_level1 = self.aggr([x_1_to_2])

        # Computing messages from Higher Order Attention Blocks Level 2
        x_0_to_0 = self.hbs_0_level2(x_0_level1, adjacency_0)
        x_1_to_1 = self.hbs_1_level2(x_1_level1, adjacency_1)
        x_2_to_2 = self.hbs_2_level2(x_2_level1, coadjacency_2)

        x_0_to_1, _ = self.hbns_0_1_level2(x_1_level1, x_0_level1, incidence_1)
        x_1_to_2, _ = self.hbns_1_2_level2(x_2_level1, x_1_level1, incidence_2)

        x_0_level2 = self.aggr([x_0_to_0])
        x_1_level2 = self.aggr([x_0_to_1, x_1_to_1])
        x_2_level2 = self.aggr([x_1_to_2, x_2_to_2])

        return x_0_level2, x_1_level2, x_2_level2
