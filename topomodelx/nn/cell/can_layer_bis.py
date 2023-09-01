"""Cellular Attention Network Layer."""
import torch
from torch.nn.parameter import Parameter

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv

# Notes:
# The attention function provided for us does not normalize the attention coefficients.
# Should this be done?
# Where should we be able to customize the non-linearities?
# Seems important for the output.
# What about the attention non-linearities do we just use what is given?
# I wanted to make this so that without attention it ends up being
# the Hodge Laplacian network.
# Maybe ask the contest organizers about this?


class CANLayer(torch.nn.Module):
    """Layer of a Cell Attention Network (CAN).

    Implementation of a layer with the cellular attention mechanism
    proposed in [GBTLSB22]_.

    Without attention this layer
    uses separate weighings for the down and up Laplacian
    in the message passing scheme proposed in [RSH22]_

    This layer is composed of one convolutional layer with an optional
    attention mechanism :
    1. Without attention the convolutional layer sends messages from
    edges to edges using the down and up Laplacian.
    2. With attention the convolution layer sends messages from
    edges to edges with attention masked by the up and down Laplacian.

    Notes
    -----
    This is the architecture proposed for entire complex classification.

    References
    ----------
    .. [GBTLSB22] Giusti et. al. Cell Attention Networks.
        https://arxiv.org/abs/2209.08179
    .. [RSH22] Rodenberry, Schaub, Hajij. Signal Processing on Cell Complexes.
        ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech
        and Signal Processing (ICASSP) 2022
        https://arxiv.org/pdf/2110.05614.pdf

    Parameters
    ----------
    channels : int
        Dimension of input features on edges (1-cells).
    activation : string
        Activation function to apply to merged message
    att : bool
        Whether to use attention.
    eps : float
        Epsilon used in the attention mechanism.
    initialization : string
        Initialization method.
    """

    def __init__(
        self,
        channels,
        activation: str = "sigmoid",
        att: bool = True,
        eps: float = 1e-5,
        initialization: str = "xavier_uniform",
    ) -> None:
        super().__init__()
        # Do I need upper and lower convolution layers? Since I think they will have different parameters
        self.conv_down = Conv(
            in_channels=channels,
            out_channels=channels,
            att=att,
            initialization=initialization,
        )
        self.conv_up = Conv(
            in_channels=channels,
            out_channels=channels,
            att=att,
            initialization=initialization,
        )
        self.conv_id = Conv(
            in_channels=channels,
            out_channels=channels,
            att=False,
            initialization=initialization,
        )
        self.aggr = Aggregation(update_func=activation)
        self.eps = eps
        self.att = att
        self.initialization = initialization
        if self.att:
            self.att_weight = Parameter(torch.Tensor(channels, 1))
        self.reset_parameters()

    def reset_parameters(self, gain: float = 1.414):
        """Reset learnable parameters.

        Parameters
        ----------
        gain : float
            Gain for the weight initialization.
        """
        self.conv_down.reset_parameters(gain=gain)
        self.conv_up.reset_parameters(gain=gain)
        self.conv_id.reset_parameters(gain=gain)
        if self.att:
            if self.initialization == "xavier_uniform":
                torch.nn.init.xavier_uniform_(self.att_weight.view(-1, 1), gain=gain)
            elif self.initialization == "xavier_normal":
                torch.nn.init.xavier_normal_(self.att_weight.view(-1, 1), gain=gain)
            else:
                raise RuntimeError(
                    "Initialization method not recognized. "
                    "Should be either xavier_uniform or xavier_normal."
                )

    def forward(self, x_1, down_laplacian, up_laplacian):
        r"""Forward pass.

        The forward pass without attention was initially proposed in [RSH22]_.
        The forward pass with attention was proposed in [GBTLSB22].

        Its equations are given in [TNN23]_ and graphically illustrated in [PSHM23]_.

        The forward pass of this layer has the following equations depending on whether attention is used.

        1. Without attention: A convolution from edges to edges using the down and up laplacian to pass messages:

        ..  math::
            \begin{align*}

            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}
            = L_{\downarrow,1} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 0 \rightarrow 1)}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}
            = L_{\uparrow,1} \cdot h_y^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 2 \rightarrow 1)}
            &🟥 \quad m_{x \rightarrow x}^{(1 \rightarrow 1)}
            = h_x^{t,(1)} \cdot \Theta^{t,(1 \rightarrow 1)}
            &🟧 \quad m_x^{(1 \rightarrow 0 \rightarrow 1)}
            = \sum_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}
            &🟧 \quad m_x^{(1 \rightarrow 2 \rightarrow 1)}
            = \sum_{y \in \mathcal{C}(x)} m_{y \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}
            &🟩: \quad m_x^{(1)}
            = m_x^{(1 \rightarrow 0 \rightarrow 1)} + m_{x \rightarrow x}^{(1 \rightarrow 1)} +m_x^{(1 \rightarrow 2 \rightarrow 1)}
            &🟦 \quad h_x^{t+1,(1)}
            = \sigma(m_{x}^{(1)})
            \end{align*}

        2. With Attention: A convolution from edges to edges using an attention mechanism masked by the down and up Laplacians:

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}
            = (L_{\uparrow,1} \odot att(h_{y \in \mathcal{L}\uparrow(x)}^{t,(1)}, h_x^{t,(1)}))_{xy} \cdot h_y^{t,(1)} \cdot
            \Theta^{t,(1 \rightarrow 2 \rightarrow 1)}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}
            = (L_{\downarrow,1} \odot att(h_{y \in \mathcal{L}\downarrow(x)}^{t,(1)}, h_x^{t,(1)}))_{xy} \cdot h_y^{t,(1)} \cdot
            \Theta^{t,(1 \rightarrow 0 \rightarrow 1)}
            &🟥 \quad m^{(1 \rightarrow 1)}_{x \rightarrow x}
            = (1+\epsilon)\cdot h_x^{t, (1)} \cdot \Theta^{t,(1 \rightarrow 1)}
            &🟧 \quad m_{x}^{(1 \rightarrow 2 \rightarrow 1)}
            = \sum_{y \in \mathcal{L}_\uparrow(x)}m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 2 \rightarrow 1)}
            &🟧 \quad m_{x}^{(1 \rightarrow 0 \rightarrow 1)}
            = \sum_{y \in \mathcal{L}_\downarrow(x)}m_{y \rightarrow \{z\} \rightarrow x}^{(1 \rightarrow 0 \rightarrow 1)}
            &🟧 \quad m^{(1 \rightarrow 1)}_{x}
            = m^{(1 \rightarrow 1)}_{x \rightarrow x}
            &🟩 \quad m_x^{(1)}
            = m_x^{(1 \rightarrow 1)} + m_{x}^{(1 \rightarrow 2 \rightarrow 1)} + m_{x}^{(1 \rightarrow 0 \rightarrow 1)}
            &🟦 \quad h_x^{t+1, (1)}
            = \sigma(\theta_{att} \cdot m_x^{(1)})\cdot \sigma(m_x^{(1)})
            \end{align*}

        References
        ----------
        .. [GBTLSB22] Giusti et. al. Cell Attention Networks.
            https://arxiv.org/abs/2209.08179
        .. [RSH22] Rodenberry, Schaub, Hajij. Signal Processing on Cell Complexes.
            ICASSP 2022 - 2022 IEEE International Conference on Acoustics, Speech and
            Signal Processing (ICASSP) 2022
            https://arxiv.org/pdf/2110.05614.pdf
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological
            Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x_1 : torch.Tensor, shape=[n_1_cells, channels]
            Input features on the edges of the cell complex.
        down_laplacian : torch.sparse
            shape=[n_1_cells, n_1_cells]
            Neighborhood matrix mapping edges to edges (L_down_1).
        up_laplacian : torch.sparse
            shape=[n_1_cells, n_1_cells]
            Neighborhood matrix mapping edges to edges (L_up_1).

        Returns
        -------
        x_1 : torch.Tensor, shape=[n_1_cells, channels]
            Output features on the edges of the cell complex.
        """
        # I don't think that the attention mechanism normalizes the attention
        # coefficients:
        #  should this be fixed? Ask the organizers
        x_down = self.conv_down(x_1, down_laplacian)
        x_up = self.conv_up(x_1, up_laplacian)
        x_id = (1 + self.eps * int(self.att)) * self.conv_id(
            x_1, torch.eye(x_1.shape[0]).to_sparse()
        )

        # The tensor diagram says to apply the non-linearities and then sum,
        # whereas the paper sums then applies the non-linearity. I followed the paper
        # here as that seems to make more sense and generalized the rodenberry paper.
        x_1 = self.aggr([x_down, x_up, x_id])

        # More attention coefficients are introduced in the CAN paper.
        # I use ELU for them because the attention mechanism in
        # message-passing uses ELU.
        if self.att:
            x_1 = x_1 * torch.nn.functional.elu(torch.mm(x_1, self.att_weight))
        return x_1
