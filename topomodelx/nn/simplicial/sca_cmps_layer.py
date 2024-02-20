"""Simplical Complex Autoencoder Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SCACMPSLayer(torch.nn.Module):
    """Layer of a Simplicial Complex Autoencoder (SCA) using the Coadjacency Message Passing Scheme (CMPS).

    Implementation of the SCA layer proposed in [1]_.

    Notes
    -----
    This is the architecture proposed for complex classification.

    Parameters
    ----------
    channels_list : list[int]
        Dimension of features at each dimension.
    complex_dim : int
        Highest dimension of chains on the input simplicial complexes.
    att : bool, default=False
        Whether to use attention.

    References
    ----------
    .. [1] Hajij, Zamzmi, Papamarkou, Maroulas, Cai.
        Simplicial complex autoencoder (2022).
        https://arxiv.org/pdf/2103.04046.pdf
    .. [2] Papillon, Sanborn, Hajij, Miolane.
        Architectures of topological deep learning: a survey on topological neural networks (2023).
        https://arxiv.org/abs/2304.10031.
    .. [3] Papillon, Sanborn, Hajij, Miolane.
        Equations of topological neural networks (2023).
        https://github.com/awesome-tnns/awesome-tnns/
    """

    def __init__(
        self,
        channels_list,
        complex_dim,
        att: bool = False,
    ) -> None:
        super().__init__()
        self.att = att
        self.dim = complex_dim
        self.channels_list = channels_list
        lap_layers = []
        inc_layers = []
        for i in range(1, complex_dim):
            conv_layer_lap = Conv(
                in_channels=channels_list[i],
                out_channels=channels_list[i],
                att=att,
            )
            conv_layer_inc = Conv(
                in_channels=channels_list[i - 1],
                out_channels=channels_list[i],
                att=att,
            )
            lap_layers.append(conv_layer_lap)
            inc_layers.append(conv_layer_inc)

        self.lap_layers = torch.nn.ModuleList(lap_layers)
        self.inc_layers = torch.nn.ModuleList(inc_layers)
        self.aggr = Aggregation(
            aggr_func="sum",
            update_func=None,
        )
        self.inter_aggr = Aggregation(
            aggr_func="mean",
            update_func="relu",
        )

    def reset_parameters(self) -> None:
        r"""Reset parameters of each layer."""
        for layer in self.lap_layers:
            if isinstance(layer, Conv):
                layer.reset_parameters()
        for layer in self.inc_layers:
            if isinstance(layer, Conv):
                layer.reset_parameters()

    def weight_func(self, x):
        r"""Weight function for intra aggregation layer according to [1]_.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        return 1 / (1 + torch.exp(-x))

    def intra_aggr(self, x):
        r"""Based on the use by [1]_.

        Parameters
        ----------
        x : torch.Tensor

        Returns
        -------
        torch.Tensor
        """
        x_list = list(torch.split(x, 1, dim=0))
        x_weight = self.aggr(x_list)
        x_weight = torch.matmul(torch.relu(x_weight), x.transpose(1, 0))
        x_weight = self.weight_func(x_weight)
        return x_weight.transpose(1, 0) * x

    def forward(self, x_list, down_lap_list, incidencet_list):
        r"""Forward pass.

        The forward pass was initially proposed in [1]_.
        Its equations are given in [3]_ and graphically illustrated in [2]_.

        Coadjacency message passing scheme:

        .. math::
            \begin{align*}
                &🟥 \quad m_{y \rightarrow x}^{(r \rightarrow r'' \rightarrow r)}
                    = M(h_{x}^{t, (r)}, h_{y}^{t, (r)},att(h_{x}^{t, (r)}, h_{y}^{t, (r)}),x,y,{\Theta^t}) \qquad \text{where } r'' < r < r'\\
                &🟥 \quad m_{y \rightarrow x}^{(r'' \rightarrow r)}
                    = M(h_{x}^{t, (r)}, h_{y}^{t, (r'')},att(h_{x}^{t, (r)}, h_{y}^{t, (r'')}),x,y,{\Theta^t})\\
                &🟧 \quad m_x^{(r \rightarrow r)}
                    = AGG_{y \in \mathcal{L}\_\downarrow(x)} m_{y \rightarrow x}^{(r \rightarrow r)}\\
                &🟧 \quad m_x^{(r'' \rightarrow r)}
                    = AGG_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}\\
                &🟩 \quad m_x^{(r)}
                    = \text{AGG}\_{\mathcal{N}\_k \in \mathcal{N}}(m_x^{(k)})\\
                &🟦 \quad h_{x}^{t+1, (r)} =
                    U(h_x^{t, (r)}, m_{x}^{(r)})
            \end{align*}

        Parameters
        ----------
        x_list : list[torch.Tensor]
            List of tensors holding the features of each chain at each level.
        down_lap_list : list[torch.Tensor]
            List of down laplacian matrices for skeletons from 1 dimension to the dimension of the simplicial complex.
        incidencet_list : list[torch.Tensor]
            List of transpose incidence matrices for skeletons from 1 dimension to the dimension of the simplicial complex.

        Returns
        -------
        list[torch.Tensor]
            Output for skeletons of each dimension (the node features are left untouched: x_list[0]).
        """
        for i in range(1, self.dim):
            x_lap = self.lap_layers[i - 1](x_list[i], down_lap_list[i - 1])
            x_inc = self.inc_layers[i - 1](x_list[i - 1], incidencet_list[i - 1])

            x_lap = self.intra_aggr(x_lap)
            x_inc = self.intra_aggr(x_inc)

            x_list[i] = self.inter_aggr([x_lap, x_inc])

        return x_list
