"""Implementation of CWN layer from Bodnar et al.: Weisfeiler and Lehman Go Cellular: CW Networks."""

import torch.nn as nn
import torch.nn.functional as F

from topomodelx.base.conv import Conv


class CWNLayer(nn.Module):
    r"""Layer of a CW Network (CWN).

    Implementation of the CWN layer proposed in [1]_.

    This module is composed of the following layers:
    1. A convolutional layer that sends messages from r-cells to r-cells.
    2. A convolutional layer that sends messages from (r-1)-cells to r-cells.
    3. A layer that creates representations in r-cells based on the received messages.
    4. A layer that updates representations in r-cells.

    Parameters
    ----------
    in_channels_0 : int
        Dimension of input features on (r-1)-cells (nodes in case r = 1).

    in_channels_1 : int
        Dimension of input features on r-cells (edges in case r = 1).

    in_channels_2 : int
        Dimension of input features on (r+1)-cells (faces in case r = 1).

    out_channels : int
        Dimension of output features on r-cells.

    conv_1_to_1 : torch.nn.Module, optional
        A module that convolves the representations of upper-adjacent neighbors of r-cells
        and their corresponding co-boundary (r+1) cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultFirstConv for more detail).

    conv_0_to_1 : torch.nn.Module, optional
        A module that convolves the representations of (r-1)-cells on the boundary of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultSecondConv for more detail).

    aggregate_fn : torch.nn.Module, optional
        A module that aggregates the representations of r-cells obtained by convolutional layers.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultAggregate for more detail).

    update_fn : torch.nn.Module, optional
        A module that updates the aggregated representations of r-cells.

        If None is passed, a default implementation of this module is used
        (check the docstring of _CWNDefaultUpdate for more detail).

    Notes
    -----
    This is the architecture proposed for entire complex classification.

    References
    ----------
    .. [1] Bodnar, et al.
        Weisfeiler and Lehman go cellular: CW networks.
        NeurIPS 2021.
        https://arxiv.org/abs/2106.12575
    """

    def __init__(
        self,
        in_channels_0,
        in_channels_1,
        in_channels_2,
        out_channels,
        conv_1_to_1=None,
        conv_0_to_1=None,
        aggregate_fn=None,
        update_fn=None,
    ) -> None:
        super().__init__()
        self.conv_1_to_1 = (
            conv_1_to_1
            if conv_1_to_1 is not None
            else _CWNDefaultFirstConv(in_channels_1, in_channels_2, out_channels)
        )
        self.conv_0_to_1 = (
            conv_0_to_1
            if conv_0_to_1 is not None
            else _CWNDefaultSecondConv(in_channels_0, in_channels_1, out_channels)
        )
        self.aggregate_fn = (
            aggregate_fn if aggregate_fn is not None else _CWNDefaultAggregate()
        )
        self.update_fn = (
            update_fn
            if update_fn is not None
            else _CWNDefaultUpdate(out_channels, out_channels)
        )

    def forward(
        self,
        x_0,
        x_1,
        x_2,
        neighborhood_1_to_1,
        neighborhood_2_to_1,
        neighborhood_0_to_1,
    ):
        r"""Forward pass.

        The forward pass was initially proposed in [1]_.
        Its equations are given in [2]_ and graphically illustrated in [3]_.

        The forward pass of this layer is composed of two convolutional steps
        that are followed by an aggregation step and a final update step.

        1. The first convolution between r-cells through (r+1)-cells exploits
        upper-adjacency neighborhood matrix and co-boundary matrix:

        ..  math::
            \begin{align*}
            &🟥 \quad m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
                = M_{\mathcal{L}\uparrow}(h_x^{t,(r)}, h_y^{t,(r)}, h_z^{t,(r')})\\
            &🟧 \quad m_x^{(r \rightarrow r' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{L}(x)} m_{y \rightarrow \{z\} \rightarrow x}^{(r \rightarrow r' \rightarrow r)}
            \end{align*}

        2. The second convolution from (r-1)-cells to r-cells exploits
        boundary neighborhood matrix:

        .. math::
            \begin{align*}
            &🟥 m_{y \rightarrow x}^{(r'' \rightarrow r)} = M_{\mathcal{B}}(h_x^{t,(r)}, h_y^{t,(r'')})\\
            &🟧 \quad m_x^{(r'' \rightarrow r)}
                = \text{AGG}_{y \in \mathcal{B}(x)} m_{y \rightarrow x}^{(r'' \rightarrow r)}
            \end{align*}

        3. Then, an aggregation step is applied:

        .. math::
            \begin{align*}
            &🟧 \quad m_x^{(r)} = AGG_{\mathcal{N}\_k \in \mathcal{N}} (m_x^k)
            \end{align*}

        4. Finally, an update step is applied:

        .. math::
            \begin{align*}
            &🟦 \quad h_x^{t+1,(r)} = U\left(h_x^{t,(r)}, m_x^{(r)}\right)
            \end{align*}

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        x_2 : torch.Tensor, shape = (n_{r+1}_cells, in_channels_{r+1})
            Input features on the (r+1)-cells.
        neighborhood_1_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        neighborhood_2_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).
        neighborhood_0_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations of the r-cells.

        References
        ----------
        .. [2] Papillon, Sanborn, Hajij, Miolane.
            Equations of topological neural networks (2023).
            https://github.com/awesome-tnns/awesome-tnns/
        .. [3] Papillon, Sanborn, Hajij, Miolane.
            Architectures of topological deep learning: a survey on topological neural networks (2023).
            https://arxiv.org/abs/2304.10031.
        """
        x_convolved_1_to_1 = self.conv_1_to_1(
            x_1, x_2, neighborhood_1_to_1, neighborhood_2_to_1
        )
        x_convolved_0_to_1 = self.conv_0_to_1(x_0, x_1, neighborhood_0_to_1)

        x_aggregated = self.aggregate_fn(x_convolved_1_to_1, x_convolved_0_to_1)
        x_updated = self.update_fn(x_aggregated, x_1)

        return x_updated


class _CWNDefaultFirstConv(nn.Module):
    r"""
    Default implementation of the first convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the first convolutional step in CWN layer.
    """

    def __init__(self, in_channels_1, in_channels_2, out_channels) -> None:
        super().__init__()
        self.conv_1_to_1 = Conv(
            in_channels_1, out_channels, aggr_norm=False, update_func=None
        )
        self.conv_2_to_1 = Conv(
            in_channels_2, out_channels, aggr_norm=False, update_func=None
        )

    def forward(self, x_1, x_2, neighborhood_1_to_1, neighborhood_2_to_1):
        r"""Forward pass.

        Parameters
        ----------
        x_1 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_2 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        neighborhood_1_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r}_cells)
            Neighborhood matrix mapping r-cells to r-cells (A_{up,r}).
        neighborhood_2_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r+1}_cells)
            Neighborhood matrix mapping (r+1)-cells to r-cells (B_{r+1}).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        x_up = F.elu(self.conv_1_to_1(x_1, neighborhood_1_to_1))
        x_coboundary = F.elu(self.conv_2_to_1(x_2, neighborhood_2_to_1))
        return x_up + x_coboundary


class _CWNDefaultSecondConv(nn.Module):
    r"""
    Default implementation of the second convolutional step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the second convolutional step in CWN layer.
    """

    def __init__(self, in_channels_0, in_channels_1, out_channels) -> None:
        super().__init__()
        self.conv_0_to_1 = Conv(
            in_channels_0, out_channels, aggr_norm=False, update_func=None
        )

    def forward(self, x_0, x_1, neighborhood_0_to_1):
        r"""Forward pass.

        Parameters
        ----------
        x_0 : torch.Tensor, shape = (n_{r-1}_cells, in_channels_{r-1})
            Input features on the (r-1)-cells.
        x_1 : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Input features on the r-cells.
        neighborhood_0_to_1 : torch.sparse, shape = (n_{r}_cells, n_{r-1}_cells)
            Neighborhood matrix mapping (r-1)-cells to r-cells (B^T_r).

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        x_boundary = F.elu(self.conv_0_to_1(x_0, neighborhood_0_to_1))
        return x_boundary


class _CWNDefaultAggregate(nn.Module):
    r"""
    Default implementation of an aggregation step in CWNLayer.

    The self.forward method of this module must be treated as
    a protocol for the aggregation step in CWN layer.
    """

    def __init__(self) -> None:
        super().__init__()

    def forward(self, x, y):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the first convolutional step.
        y : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Representations on the r-cells produced by the second convolutional step.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Aggregated representations on the r-cells.
        """
        return x + y


class _CWNDefaultUpdate(nn.Module):
    r"""Default implementation of an update step in CWNLayer."""

    def __init__(self, in_channels, out_channels) -> None:
        super().__init__()
        self.transform = nn.Linear(in_channels, out_channels)

    def forward(self, x, x_prev=None):
        r"""Forward pass.

        Parameters
        ----------
        x : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            New representations on the r-cells obtained after the aggregation step.
        x_prev : torch.Tensor, shape = (n_{r}_cells, in_channels_{r})
            Original representations on the r-cells passed into the CWN layer.

        Returns
        -------
        torch.Tensor, shape = (n_{r}_cells, out_channels)
            Updated representations on the r-cells.
        """
        x_updated = F.elu(self.transform(x))
        return x_updated
