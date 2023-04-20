import torch
import torch.nn as nn
import torch.nn.functional as F
from topomodelx.layers.message_passing import HigherOrderMessagePassing


class CXINConv(HigherOrderMessagePassing):
    r"""

    Example :
        from toponetx.simplicial_complex import SimplicialComplex
        from topomodelx.util.tensors_util import coo_2_torch_tensor


        SC= SimplicialComplex([[0,1],[1,2]])

        A0 = coo_2_torch_tensor(SC.get_higher_order_adj(0))

        A1=  coo_2_torch_tensor(SC.get_higher_order_coadj(1))

        B1 = coo_2_torch_tensor(SC.get_boundary_operator(1))

        x_v = torch.rand(3,10)

        x_e = torch.rand(2,10)


        model = CXINConv (10,20)

        print(model(x_v,B1.t()))

    """

    def __init__(
        self,
        in_channels,
        out_channels,
        mlp_hidden=2,
        mlp_activation=nn.ReLU(),
        message_type="sender",
        epsilon=None,
        mlp_batchnorm=False,
        aggregate="sum",
        kernel_initializer="xavier_uniform",
        bias_initializer="zeros",
    ):
        super(CXINConv, self).__init__(aggregate=aggregate)

        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.message_type = message_type
        self.init_scheme = kernel_initializer
        self.epsilon = epsilon

        layers = []
        self.mlp = nn.Sequential()
        layers.append(nn.Linear(in_channels, out_channels))
        layers.append(mlp_activation)

        if self.mlp_batchnorm is not False:
            layers.append(nn.BatchNorm1d(out_channels))

        for _ in range(0, mlp_hidden):
            layers.append(nn.Linear(out_channels, out_channels))
            layers.append(mlp_activation)
            if self.mlp_batchnorm is not False:
                layers.append(nn.BatchNorm1d(out_channels))
        self.mlp = nn.Sequential(*layers)

        if self.epsilon is None:
            self.epsilon = torch.nn.Parameter(torch.Tensor(1))
        else:
            torch.nn.Parameter(torch.Tensor([epsilon]), requires_grad=False)

        self.reset_parameters()

    def reset_parameters(self):
        if self.epsilon is None:
            nn.init.zeros_(self.eps)

    def forward(self, x, a, aggregate_sign=True, aggregate_value=True):
        if self.message_type == "sender":
            output = self.mlp(
                self.propagate(x, a, aggregate_sign, aggregate_value)
                + (1 + self.epsilon) * a.matmul(x)
            )  # a sends x to a*x
        elif self.message_type == "receiver":
            # a must be symmetric
            output = self.mlp(
                self.propagate(x, a, aggregate_sign) + (1 + self.epsilon) * x
            )  # a collects neighbors of x
        else:
            raise RuntimeError(
                f" message type "
                f"'{self.message_type}' is not supported."
                + "Supported types are [sender,receiver]"
            )
        return output


class CXINMerge(nn.Module):
    r"""
    Example :
        from toponetx.simplicial_complex import SimplicialComplex
        from topomodelx.util.tensors_util import coo_2_torch_tensor

        SC= SimplicialComplex([[0,1],[1,2]])

        A0 = coo_2_torch_tensor(SC.get_higher_order_adj(0))

        A1=  coo_2_torch_tensor(SC.get_higher_order_coadj(1))

        B1 = coo_2_torch_tensor(SC.get_boundary_operator(1))

        x_v = torch.rand(3,10)

        x_e = torch.rand(2,10)

        model =CXINMerge(10,20)

        out_e = model(x_v, x_e, B1.t(), A1 )

        print(out_e)
    """

    def __init__(
        self,
        in_ch_1,
        in_ch_2,
        out_channels,
        mlp_hidden=2,
        mlp_activation=nn.ReLU(),
        message_type="sender",
        epsilon=None,
        mlp_batchnorm=False,
        aggregate="sum",
        kernel_initializer="xavier_uniform",
        bias_initializer="zeros",
        merge_type="conc",
    ):
        super(CXINMerge, self).__init__()

        self.merge_type = merge_type

        self.CXINConv1 = CXINConv(
            in_ch_1,
            out_channels,
            mlp_hidden,
            mlp_activation,
            message_type,
            epsilon,
            mlp_batchnorm,
            aggregate,
            kernel_initializer,
            bias_initializer,
        )

        self.CXINConv2 = CXINConv(
            in_ch_2,
            out_channels,
            mlp_hidden,
            mlp_activation,
            message_type,
            epsilon,
            mlp_batchnorm,
            aggregate,
            kernel_initializer,
            bias_initializer,
        )

        if self.merge_type == "conc":

            self.merger = nn.Linear(2 * out_channels, out_channels)

    def forward(self, x1, x2, G1, G2):
        if G1.shape[0] != G2.shape[0]:
            raise ValueError(
                f"Input operators G1 and G2 must have the same target, got {G1.shape[0]} and {G2.shape[0]}"
            )

        out1 = self.CXINConv1(x1, G1)
        out2 = self.CXINConv2(x2, G2)
        if self.merge_type == "conc":
            output = self.merger(torch.cat((out1, out2), 1))
        elif self.merge_type == "sum":
            return out1 + out2
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.merge_type}' is not supported"
            )

        return output


class CXINMergeToTarget(HigherOrderMessagePassing):
    r"""
    Example :
        from toponetx.simplicial_complex import SimplicialComplex
        from topomodelx.util.tensors_util import coo_2_torch_tensor

        SC= SimplicialComplex([[0,1],[1,2]])

        A0 = coo_2_torch_tensor(SC.get_higher_order_adj(0))

        A1=  coo_2_torch_tensor(SC.get_higher_order_coadj(1))

        B1 = coo_2_torch_tensor(SC.get_boundary_operator(1))

        x_v = torch.rand(3,3) # target

        x_e = torch.rand(2,5) # source

        model =  CXINMergeToTarget(3,5)

        out_e = model(x_e,x_v,B1.t())

        print(out_e)
    """

    def __init__(
        self,
        in_channel,
        out_channel,
        mlp_hidden=2,
        mlp_activation=nn.ReLU(),
        epsilon=0,
        mlp_batchnorm=False,
        aggregate="sum",
        kernel_initializer="xavier_uniform",
        bias_initializer="zeros",
    ):
        super(CXINMergeToTarget, self).__init__(aggregate=aggregate)

        self.out_channel = out_channel
        self.mlp_activation = mlp_activation
        self.mlp_batchnorm = mlp_batchnorm
        self.init_scheme = kernel_initializer
        self.epsilon = epsilon

        layers = []
        self.mlp = nn.Sequential()
        layers.append(nn.Linear(out_channel, out_channel))
        layers.append(mlp_activation)

        if self.mlp_batchnorm is not False:
            layers.append(nn.BatchNorm1d(out_channel))

        for _ in range(0, mlp_hidden):
            layers.append(nn.Linear(out_channel, out_channel))
            layers.append(mlp_activation)
            if self.mlp_batchnorm is not False:
                layers.append(nn.BatchNorm1d(out_channel))
        self.mlp = nn.Sequential(*layers)

        self.weight = torch.nn.Parameter(torch.Tensor(in_channel, out_channel))

        if self.epsilon is None:
            self.epsilon = torch.nn.Parameter(torch.Tensor(1))
        else:
            self.epsilon = epsilon

        self.reset_parameters()

    def reset_parameters(self, gain=1.414):
        if self.epsilon is None:
            nn.init.zeros_(self.eps)
        if self.init_scheme == "xavier_uniform":

            nn.init.xavier_uniform_(self.weight, gain=gain)

        elif self.init_scheme == "xavier_normal":

            nn.init.xavier_normal_(self.weight, gain=gain)

        elif self.init_scheme == "uniform":

            stdv = 1.0 / torch.sqrt(self.weight.size(1))
            self.weight.data.uniform_(-stdv, stdv)
            if self.bias is not None:
                self.bias.data.uniform_(-stdv, stdv)
        else:
            raise RuntimeError(
                f" weight initializer " f"'{self.init_scheme}' is not supported"
            )

    def forward(self, x_target, x_source, a, aggregate_sign=True):

        assert x_target.shape[-1] == self.out_channel

        if (
            x_target.shape[-1] != x_source.shape[-1]
        ):  # make dimensions the same as the target
            support = x_source @ self.weight
        else:
            support = x_source
        output = self.mlp(
            self.propagate(support, a, aggregate_sign) + (1 + self.epsilon) * x_target
        )  # a collects neighbors of x

        return output


class CXINGeneral(nn.Module):
    r"""
    Example :
        from toponetx.simplicial_complex import SimplicialComplex
        from topomodelx.util.tensors_util import coo_2_torch_tensor

        SC= SimplicialComplex([[0,1],[1,2]])

        A0 = coo_2_torch_tensor(SC.get_higher_order_adj(0))

        A1=  coo_2_torch_tensor(SC.get_higher_order_coadj(1))

        B1 = coo_2_torch_tensor(SC.get_boundary_operator(1))

        x_v_1 = torch.rand(3,3)
        x_v_2 = torch.rand(3,4)

        x_e = torch.rand(2,10)

        model = CXINGeneral([3,4],10)
        out_e = model (x_e,[x_v_1,x_v_2],[B1.t(),B1.t() ])
        print(out_e.shape)
    """

    def __init__(
        self,
        in_ch_list,
        out_channel,
        mlp_hidden=2,
        mlp_activation=nn.ReLU(),
        epsilon=None,
        mlp_batchnorm=False,
        aggregate="sum",
        kernel_initializer="xavier_uniform",
        bias_initializer="zeros",
        merge_type="conc",
    ):
        super(CXINGeneral, self).__init__()

        assert isinstance(in_ch_list, list)

        assert isinstance(out_channel, int)

        if merge_type not in ["conc", "sum"]:
            raise RuntimeError(f"merge type must be in [conc,sum], got {merge_type}")

        self.in_ch_list = in_ch_list
        self.out_channel = out_channel
        self.merge_type = merge_type
        self.CXIN_list = nn.ModuleList(
            [
                CXINMergeToTarget(
                    i,
                    out_channel,
                    mlp_hidden,
                    mlp_activation,
                    epsilon,
                    mlp_batchnorm,
                    aggregate,
                    kernel_initializer,
                    bias_initializer,
                )
                for i in in_ch_list
            ]
        )

        if merge_type == "conc":
            self.merger = nn.Linear(len(in_ch_list) * out_channel, out_channel)

    def forward(self, x_target, x_neighbor_list, G_list):

        assert len(x_neighbor_list) == len(G_list)

        assert x_target.shape[-1] == self.out_channel

        all_tensors = [
            self.CXIN_list[i](x_target, x_neighbor_list[i], G_list[i])
            for i in range(0, len(x_neighbor_list))
        ]

        if self.merge_type == "conc":
            output = self.merger(torch.cat(tuple(all_tensors), 1))
        elif self.merge_type == "sum":
            output = torch.stack(all_tensors, axis=0).sum(axis=0)
        else:
            raise RuntimeError(" merge type must be in [conc,sum]")

        return output
