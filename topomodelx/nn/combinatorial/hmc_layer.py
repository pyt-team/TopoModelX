import torch
import toponetx as tnx
from topomodelx.base.aggregation import Aggregation
from typing import List

from topomodelx.base.hbs import HBS
from topomodelx.base.hbns import HBNS


class HMCLayer(torch.nn.Module):
    def __init__(self, in_channels: List[int], intermediate_channels: List[int], out_channels: List[int],
                 negative_slope: float, softmax_attention=False, update_func_attention=None,
                 update_func_aggregation=None, initialization="xavier_uniform"):

        super().__init__()

        assert len(in_channels) == 3 and len(intermediate_channels) == 3 and len(out_channels) == 3

        in_channels_0, in_channels_1, in_channels_2 = in_channels
        intermediate_channels_0, intermediate_channels_1, intermediate_channels_2 = intermediate_channels
        out_channels_0, out_channels_1, out_channels_2 = out_channels

        self.hbs_0_level1 = HBS(source_in_channels=in_channels_0, source_out_channels=intermediate_channels_0,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_0_1_level1 = HBNS(source_in_channels=in_channels_0,
                                    source_out_channels=intermediate_channels_0,
                                    target_in_channels=in_channels_1,
                                    target_out_channels=intermediate_channels_1,
                                    negative_slope=negative_slope, softmax=softmax_attention, update_func=update_func_attention, initialization=initialization)

        self.hbns_1_2_level1 = HBNS(source_in_channels=in_channels_1,
                                    source_out_channels=intermediate_channels_1,
                                    target_in_channels=in_channels_2,
                                    target_out_channels=intermediate_channels_2,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbs_0_level2 = HBS(source_in_channels=intermediate_channels_0, source_out_channels=out_channels_0,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_0_1_level2 = HBNS(source_in_channels=intermediate_channels_0,
                                    source_out_channels=out_channels_0,
                                    target_in_channels=intermediate_channels_1,
                                    target_out_channels=out_channels_1,
                                    negative_slope=negative_slope,
                                    softmax=softmax_attention,
                                    update_func=update_func_attention,
                                    initialization=initialization)

        self.hbs_1_level2 = HBS(source_in_channels=intermediate_channels_1, source_out_channels=out_channels_1,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.hbns_1_2_level2 = HBNS(source_in_channels=intermediate_channels_1,
                                    source_out_channels=out_channels_1,
                                    target_in_channels=intermediate_channels_2,
                                    target_out_channels=out_channels_2,
                                    negative_slope=negative_slope, softmax=softmax_attention,
                                    update_func=update_func_attention, initialization=initialization)

        self.hbs_2_level2 = HBS(source_in_channels=intermediate_channels_2, source_out_channels=out_channels_2,
                                negative_slope=negative_slope, softmax=softmax_attention,
                                update_func=update_func_attention, initialization=initialization)

        self.aggr = Aggregation(aggr_func="sum", update_func=update_func_aggregation)

    def forward(self, x_0, x_1, x_2, adjacency_0, adjacency_1, coadjacency_2, incidence_1, incidence_2):
        # Computing messages from Higher Order Attention Blocks Level 1
        x_0_to_0 = self.hbs_0_level1(x_0, adjacency_0)
        x_1_to_0, x_0_to_1 = self.hbns_0_1_level1(x_0, x_1, incidence_1)
        x_2_to_1, x_1_to_2 = self.hbns_1_2_level1(x_1, x_2, incidence_2)

        x_0_level1 = self.aggr([x_0_to_0, x_1_to_0])
        x_1_level1 = self.aggr([x_0_to_1, x_2_to_1])
        x_2_level1 = x_1_to_2

        # Computing messages from Higher Order Attention Blocks Level 2
        x_0_to_0 = self.hbs_0_level2(x_0_level1, adjacency_0)
        x_1_to_1 = self.hbs_1_level2(x_1_level1, adjacency_1)
        x_2_to_2 = self.hbs_2_level2(x_2_level1, coadjacency_2)
        _, x_0_to_1 = self.hbns_0_1_level2(x_0_level1, x_1_level1, incidence_1)
        _, x_1_to_2 = self.hbns_1_2_level2(x_1_level1, x_2_level1, incidence_2)

        x_0_level2 = x_0_to_0
        x_1_level2 = self.aggr([x_0_to_1, x_1_to_1])
        x_2_level2 = self.aggr([x_1_to_2, x_2_to_2])

        return x_0_level2, x_1_level2, x_2_level2