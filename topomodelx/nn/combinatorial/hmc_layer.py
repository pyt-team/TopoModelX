import torch
import toponetx as tnx
from topomodelx.base.conv import Conv
from topomodelx.base.aggregation import Aggregation
import torch.nn.functional as F
from typing import List


class HMCLayer(torch.nn.Module):
    def __init__(self, in_channels: List[int], intermediate_channels: List[int], out_channels: List[int]):
        super().__init__()
        assert len(in_channels) == 3 and len(intermediate_channels) == 3 and len(out_channels) == 3
        in_channels_0, in_channels_1, in_channels_2 = in_channels
        intermediate_channels_0, intermediate_channels_1, intermediate_channels_2 = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.conv_level1_0_to_0 = Conv(
            in_channels=in_channels_0, out_channels=intermediate_channels_0, att=True
        )
        self.conv_level1_0_to_1 = Conv(
            in_channels=in_channels_0, out_channels=intermediate_channels_1, att=True
        )
        self.conv_level1_1_to_0 = Conv(
            in_channels=in_channels_1, out_channels=intermediate_channels_0, att=True
        )
        self.conv_level1_1_to_2 = Conv(
            in_channels=in_channels_1, out_channels=intermediate_channels_2, att=True
        )
        self.conv_level1_2_to_1 = Conv(
            in_channels=in_channels_2, out_channels=intermediate_channels_1, att=True
        )
        self.conv_level2_0_to_0 = Conv(
            in_channels=intermediate_channels_0, out_channels=out_channels_0, att=True
        )
        self.conv_level2_0_to_1 = Conv(
            in_channels=intermediate_channels_0, out_channels=out_channels_1, att=True
        )
        self.conv_level2_1_to_1 = Conv(
            in_channels=intermediate_channels_1, out_channels=out_channels_1, att=True
        )
        self.conv_level2_1_to_2 = Conv(
            in_channels=intermediate_channels_1, out_channels=out_channels_2, att=True
        )
        self.conv_level2_2_to_2 = Conv(
            in_channels=intermediate_channels_2, out_channels=out_channels_2, att=True
        )

        self.aggr = Aggregation(aggr_func="sum", update_func="sigmoid")

        def forward(x_0, x_1, x_2, adjacence_0, adjacence_1, coadjacence_2, incidence_1, incidence_2):
            incidence_1_transpose = incidence_1.to_dense().T.to_sparse()
            incidence_2_transpose = incidence_2.to_dense().T.to_sparse()

            x_0_level1 = self.aggr([self.conv_level1_0_to_0(x_0, adjacence_0),
                                    self.conv_level1_0_to_1(x_1, incidence_1),
                                    ])
            x_1_level1 = self.aggr([self.conv_level1_0_to_1(x_0, incidence_1_transpose),
                                    self.conv_level1_2_to_1(x_2, incidence_2)])

            x_2_level1 = self.conv_level1_1_to_2(x_1, incidence_2_transpose)

            x_0_level2 = self.conv_level2_0_to_0(x_0_level1, adjacence_0)

            x_1_level2 = self.aggr([self.conv_level2_0_to_1(x_0_level1, incidence_1_transpose),
                                    self.conv_level2_1_to_1(x_1_level1, adjacence_1)])

            x_2_level2 = self.aggr([self.conv_level2_1_to_2(x_1_level1, incidence_2_transpose),
                                    self.conv_level2_2_to_2(x_2_level1, coadjacence_2)])

            return x_0_level2, x_1_level2, x_2_level2
