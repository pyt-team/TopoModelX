"""Higher Order Attention Block for squared neighborhoods (HBS) for message passing module."""

from multiprocessing import Pool

import numpy as np
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


class HBS(MessagePassing):
    r"""Higher Order Attention Block layer for squared neighborhoods (HBS). HBS layers were introduced in [HAJIJ23]_, Definitions 31 and 32.

    Let :math:`\mathcal{X}` be a combinatorial complex, we denote :math:`\mathcal{C}^k(\mathcal{X}, \mathbb{R}^d)` as the :math:`d`-dimensional
    :math:`\mathbb{R}`-valued vector space of signals over :math:`\Sigma^k`, the :math:`k`-th skeleton of :math:`\mathcal{X}` subject to a certain total order.
    Elements of this space are called :math:`k`-cochains of :math:`\mathcal{X}`.
    If :math:`d = 1`, we denote :math:`\mathcal{C}^k(\mathcal{X})`.

    Let :math:`N: \mathcal{C}^s(\mathcal{X}) \rightarrow \mathcal{C}^s(\mathcal{X})` be a cochain map endomorphism of the space of signals over
    :math:`\Sigma^s` of \mathcal{X}. The matrix representation of :math:`N` has shape :math:`n_{cells} \times n_{cells}`, where
    :math:`n_{cells}` denotes the cardinality of :math:`\Sigma^s`.

    The higher order attention block induced by :math:`N` is the cochain map

    ..  math::
        \begin{align}
            HBS_N: \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{in}}}) \rightarrow \mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{out}}}),
        \end{align}

    where :math:`d^{s_{in}}` and :math:`d^{s_{out}}` are the input and output dimensions of the HBS block, respectively,
    also denoted as source_in_channels and source_out_channels, respectively.

    :math:`HBS_N` is defined by

    ..  math::
        \phi(\sum_{p=1}^{\text{m\_hop}}(N^p \odot A_p) X W_p )

    where :math:`X` is the cochain matrix representation of shape [n_cells, source_in_channels] under the canonical basis
    of :math:`\mathcal{C}^s(\mathcal{X},\mathbb{R}^{d^{s_{in}}})`, induced by the total order of :math:`\Sigma^s`, that contains
    the input features for each cell. The :math:`\odot` symbol denotes the Hadamard product, namely the entry-wise product, and
    :math:`\phi` is a non-linear activation function. :math:`W_p` is a learnable weight matrix of shape [source_in_channels, source_out_channels]
    for each :math:`p`, and :math:`A_p` is an attention matrix with the same dimensionality as the input neighborhood matrix :math:`N`, i.e., [n_cells, n_cells].
    The indices :math:`(i,j)` of the attention matrix :math:`A_p` are computed as

    ..  math::
        A_p(i,j) = \frac{e_{i,j}^p}{\sum_{k=1}^{columns(N)} e_{i,k}^p}

    where

    ..  math::
        e_{i,j}^p = S(\text{LeakyReLU}([X_iW_p||X_jW_p]a_p))

    and where || denotes concatenation, :math:`a_p` is a learnable column vector of length :math:`2*source_out_channels`, and :math:`S` is
    the exponential function if softmax is used and the identity function otherwise.

    This HBS class just contains the sparse implementation of the block.

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
    negative_slope : float
        Negative slope of the LeakyReLU activation function.
    softmax : bool, optional
        Whether to use softmax in the computation of the attention matrix. Default is False.
    m_hop : int, optional
        Maximum number of hops to consider in the computation of the layer function. Default is 1.
    update_func : {None, 'sigmoid', 'relu', 'tanh}, optional
        Activation function :math:`phi` in the computation of the output of the layer.
        If None, :math:`phi` is the identity function. Default is None.
    initialization : {'xavier_uniform', 'xavier_normal'}, optional
        Initialization method for the weights of W_p and :math:`a_p`. Default is 'xavier_uniform'.
    """

    def __init__(
        self,
        source_in_channels,
        source_out_channels,
        negative_slope,
        softmax=False,
        m_hop=1,
        update_func=None,
        initialization="xavier_uniform",
    ):

        super().__init__(
            att=True,
            initialization=initialization,
        )

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

    def get_device(self):
        """Get the device on which the layer's learnable parameters are stored."""
        return self.weight[0].device

    def reset_parameters(self, gain=1.414):
        r"""Reset learnable parameters.

        Parameters
        ----------
        gain : float, optional
            Gain for the weight initialization. Default is 1.414.
        """

        def reset_specific_hop_parameters(weight, att_weight):
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
            reset_specific_hop_parameters(w, a)

    def update(self, message):
        r"""Update signal features on each cell with an activation function, either sigmoid, ReLU or tanh.

        Parameters
        ----------
        message : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features before the activation function.

        Returns
        -------
        _ : torch.Tensor, shape=[n_cells, out_channels]
            Output signal features after the activation function :math:`\phi`.
        """
        if self.update_func == "sigmoid":
            return torch.sigmoid(message)
        elif self.update_func == "relu":
            return torch.nn.functional.relu(message)
        elif self.update_func == "tanh":
            return torch.nn.functional.tanh(message)

    def attention(self, message, A_p, a_p):
        """Compute attention matrix.

        Parameters
        ----------
        message : torch.Tensor, shape [n_messages, source_out_channels]
            Message tensor. This is the result of the matrix multiplication of the cochain matrix X with the weight
            matrix W_p.
        A_p : torch.sparse, shape [n_cells, n_cells]
            Neighborhood matrix to the power p.
        a_p : torch.Tensor, shape [2*source_out_channels, 1]
            Attention weight vector.

        Returns
        -------
        att_p : torch.sparse, shape [n_messages, n_messages]. Represents the attention matrix A_p.
        """
        n_messages = message.shape[0]
        source_index_i, source_index_j = A_p._indices()
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

    def forward(self, x_source, neighborhood):
        """Forward pass.

        The forward pass of the Higher Order Attention Block layer. x_source is the cochain matrix X used as input
        features for the layer. neighborhood is the neighborhood matrix A. Note that the neighborhood matrix shape
        should be [n_cells, n_cells] where n_cells is the number of rows in x_source.

        Parameters
        ----------
        x_source : torch.Tensor, shape=[n_cells, source_in_channels]
            Cochain matrix X used as input of the layer.
        neighborhood : torch.sparse, shape=[n_cells, n_cells]
            Neighborhood matrix used to compute the HBS layer.

        Returns
        -------
        _ : Tensor, shape=[n_cells, source_out_channels]
            Output features of the layer.
        """
        message = [
            torch.mm(x_source, w) for w in self.weight
        ]  # [m-hop, n_source_cells, d_t_out]
        # Create a torch.eye with the device of x_source
        result = torch.eye(x_source.shape[0], device=self.get_device()).to_sparse_coo()
        neighborhood = [
            result := torch.sparse.mm(neighborhood, result) for _ in range(self.m_hop)
        ]

        att = [
            self.attention(m_p, A_p, a_p)
            for m_p, A_p, a_p in zip(message, neighborhood, self.att_weight)
        ]

        def sparse_hadamard(A_p, att_p):
            return torch.sparse_coo_tensor(
                indices=A_p.indices(),
                values=att_p.values() * A_p.values(),
                size=A_p.shape,
                device=self.get_device(),
            )

        neighborhood = [
            sparse_hadamard(A_p, att_p) for A_p, att_p in zip(neighborhood, att)
        ]
        message = [torch.mm(n_p, m_p) for n_p, m_p in zip(neighborhood, message)]
        result = torch.zeros_like(message[0])

        for m_p in message:
            result += m_p
        if self.update_func is None:
            return result

        return self.update(result)
