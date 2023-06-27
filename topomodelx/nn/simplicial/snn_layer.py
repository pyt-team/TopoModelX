"""Simplicial Neural Network Layer."""
import torch

from topomodelx.base.aggregation import Aggregation
from topomodelx.base.conv import Conv


class SNNLayer(torch.nn.Module):
    """Layer of a Simplicial Neural Network (SNN).

    Implementation of the SNN layer proposed in [SNN20].


    References
    ----------
    .. [SNN20] Stefania Ebli, Michael Defferrard and Gard Spreemann.
        Simplicial Neural Networks.
        Topological Data Analysis and Beyond workshop at NeurIPS.
        https://arxiv.org/abs/2010.03633

    Parameters
    ----------
    K : int
        Maximum polynomial degree for Laplacian.
    in_channels : int
        Dimension of features on each simplicial cell.
    out_channels : int
        Dimension of output representation on each simplicial cell.
    initialization : string
        Initialization method.
    """

    def __init__(self, in_channels, out_channels, K):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.K = K

        convs = [
            Conv(in_channels=in_channels, out_channels=out_channels, update_func="relu")
            for _ in range(self.K)
        ]

        self.convs = torch.nn.ModuleList(convs)

        self.aggr = Aggregation(aggr_func="sum", update_func="relu")

    def reset_parameters(self):
        r"""Reset learnable parameters."""

        for conv in self.convs:
            conv.reset_parameters()

    def forward(self, x, laplacian):
        r"""Forward pass.

        The forward pass was initially proposed in [SNN20]_.
        Its equations are adapted from [TNN23]_,  graphically illustrated in [PSHM23]_.

        .. math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow x}^{p, (d \rightarrow d)}  = ((H_{d})^p)\_{xy} \cdot h_y^{t,(d)} \cdot \Theta^{t,p}\\
            &🟧 \quad m_{x}^{p, (d \rightarrow d)}  = \sum_{y \in (\mathcal{L}\_\uparrow + \mathcal{L}\_\downarrow)(x)} m_{y \rightarrow x}^{p, (d \rightarrow d)}\\
            &🟧 \quad m_x^{(d \rightarrow d)}  = \sum_{p=1}^{P_1} (m_{x}^{p,(d \rightarrow d)})^{p}\\
            &🟩 \quad m_x^{(d)}  = m_x^{(d \rightarrow d)}\\
            &🟦 \quad h_x^{t+1, (d)}  = \sigma (m_{x}^{(d)})
            \end{align*}

        References
        ----------
        .. [SNN20] Stefania Ebli, Michael Defferrard and Gard Spreemann.
            Simplicial Neural Networks.
            Topological Data Analysis and Beyond workshop at NeurIPS.
            https://arxiv.org/abs/2010.03633
        .. [TNN23] Equations of Topological Neural Networks.
            https://github.com/awesome-tnns/awesome-tnns/
        .. [PSHM23] Papillon, Sanborn, Hajij, Miolane.
            Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.
            (2023) https://arxiv.org/abs/2304.10031.

        Parameters
        ----------
        x: torch.Tensor, shape=[n_simplices, in_channels]
            Input features on the simplices of the simplicial complex for the given simplicial degree.
        laplacian : torch.sparse, shape=[n_simplices, n_simplices]
            Simplicial Laplacian matrix for the given simplicial degree.

        Returns
        -------
        _ : torch.Tensor, shape=[n_simplices, out_channels]
            Output features on the nodes of the simplicial complex.
        """

        outputs = []
        laplacian_power = torch.eye(laplacian.shape[0])
        outputs.append(self.convs[0](x, laplacian_power))

        for i in range(1, self.K):
            laplacian_power = torch.mm(laplacian_power, laplacian)

            outputs.append(self.convs[i](x, laplacian_power))

        x = self.aggr(outputs)
        return x
