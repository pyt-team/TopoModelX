"""Higher Order Attention Network (HOAN) layer."""
import torch

from topomodelx.base.conv import Conv


class HOANLayer(torch.nn.Module):
    """HOAN layer.

    Implements a HOAN layer.

    Parameters
    ----------
    in_channels : int
        Dimension of input features.
    out_channels : int
        Dimension of output features.
    """

    def __init__(
        self,
        in_channels_source,
        out_channels_source,
        in_channels_target,
        out_channels_target,
        dropout=0.2,
    ):
        super().__init__()

        self.in_channels_source = in_channels_source
        self.out_channels_source = out_channels_source
        self.in_channels_target = in_channels_target
        self.out_channels_target = out_channels_target
        self.dropout = dropout

        self.w_source = torch.nn.Parameter(
            torch.empty(in_channels_source, out_channels_target)
        )
        self.w_target = torch.nn.Parameter(
            torch.empty(in_channels_target, out_channels_source)
        )
        self.a = torch.nn.Parameter(
            torch.empty(out_channels_target + out_channels_source, 1)
        )
        self.bias_source = torch.nn.Parameter(torch.empty(out_channels_target))
        self.bias_target = torch.nn.Parameter(torch.empty(out_channels_source))
        self.leaky_relu = torch.nn.LeakyReLU()

        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        """Reset learnable parameters."""
        torch.nn.init.xavier_uniform_(self.w_source.data, gain=gain)
        torch.nn.init.xavier_uniform_(self.w_target.data, gain=gain)
        torch.nn.init.xavier_uniform_(self.a.data, gain=gain)
        self.bias_source.data.fill_(0)
        self.bias_target.data.fill_(0)

    def forward(self, x_source, x_target, adjacency):
        """Forward computation.

        Parameters
        ----------
        x_1 : torch.Tensor, shape=[n_edges, in_channels]
            Input features on the edges of the simplicial complex.
        incidence_1 : torch.sparse
            shape=[n_nodes, n_edges]
            Incidence matrix mapping edges to nodes (B_1).

        Returns
        -------
        x_1 : torch.Tensor, shape=[n_edges, out_channels]
            Output features on the edges of the simplicial complex.
        """
        x_source = torch.nn.functional.F.dropout(
            x_source, self.dropout, training=self.training
        )
        xw_source = x_source @ self.w_source

        if x_target is not None:
            xw_target = x_target @ self.w_target
        else:
            xw_target = None

        if x_target is not None:
            x_target = torch.nn.functional.F.dropout(
                x_target, self.dropout, training=self.training
            )
            xw_target = x_target @ self.w_target

            e = self.leaky_relu(
                torch.cat(
                    (
                        self.a[: self.out_channels_source, :].T @ xw_source,
                        self.a[self.out_channels_source :, :].T @ xw_target,
                    ),
                    dim=1,
                )
            )
            f = self.leaky_relu(
                torch.cat(
                    (
                        self.a[: self.our_channels_target, :].T @ xw_target,
                        self.a[self.out_channels_target :, :].T @ xw_source,
                    ),
                    dim=1,
                )
            )
        else:
            e = self.leaky_relu(
                torch.cat(
                    (
                        self.a[: self.out_channels_source, :].T @ xw_source,
                        self.a[self.out_channels_source :, :].T @ xw_source,
                    ),
                    dim=1,
                )
            )

        zeros = -10e12 * torch.ones_like(e)
        attention_source = torch.nn.functional.F.softmax(
            torch.where(adjacency > 0, e, zeros), dim=1
        )
        attention_source = torch.nn.functional.F.dropout(
            attention_source, self.dropout, training=self.training
        )
        update_source = torch.nn.functional.F.elu(
            attention_source @ xw_source + self.bias_source
        )

        if x_target is not None:
            attention_target = torch.nn.functional.F.softmax(
                torch.where(adjacency.T > 0, f, zeros), dim=1
            )
            attention_target = torch.nn.functional.F.dropout(
                attention_target, self.dropout, training=self.training
            )
            update_target = torch.nn.functional.F.elu(
                attention_target @ xw_target + self.bias_target
            )
        else:
            update_target = None

        return update_source, update_target
