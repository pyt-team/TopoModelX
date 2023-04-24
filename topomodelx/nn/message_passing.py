"""
@author: Mustafa Hajij
"""

from typing import Optional, Tuple

import torch
from modelnetx.scatter.scatter import scatter
from torch import Tensor


class HigherOrderMessagePassing(torch.nn.Module):

    """
    Example
        from toponetx.simplicial_complex import SimplicialComplex
        from modelnetx.util.tensors_util import coo_2_torch_tensor
        from modelnetx.nn.message_passing import HigherOrderMessagePassing
        SC= SimplicialComplex([[0,1],[1,2]])
        B1 = coo_2_torch_tensor(SC.incidence_matrix(1))
        homp = HigherOrderMessagePassing()
        x_e = torch.rand(2,10)
        x_v = homp(x_e,B1)


        print(x_v)


        A1 = coo_2_torch_tensor(SC.get_higher_order_coadj(1))
        x_e = torch.rand(2,10)
        x_e = homp(x_e,A1)

    """

    def __init__(self, aggregate="sum"):
        super().__init__()
        self.agg = scatter(aggregate)

    def forward(
        self, x: Tensor, a: Tensor, aggregate_sign=True, aggregate_value=True
    ) -> Tensor:
        return self.propagate(x, a, aggregate_sign, aggregate_value)

    def propagate(self, x, a, aggregate_sign=True, aggregate_value=True) -> Tensor:
        assert isinstance(x, Tensor)
        assert isinstance(a, Tensor)
        assert len(a.shape) == 2

        target, source = a.coalesce().indices()

        self.index_i = target  # Ni, target
        self.index_j = source  # Nj, source
        # each source cell sends messages
        # to its "neighbors" according to Nj
        # defined the operator a.
        messages = self.message(x)
        if aggregate_sign and aggregate_value:
            values = a.coalesce().values()
            messages = torch.mul(values.unsqueeze(-1), messages)

        elif aggregate_sign and not aggregate_value:
            sign = torch.sign(a.coalesce().values())
            messages = torch.mul(sign.unsqueeze(-1), messages)

        elif aggregate_value and not aggregate_sign:
            values = a.coalesce().values()
            messages = torch.mul(torch.abs(values.unsqueeze(-1)), messages)

        # each targer cell aggregates
        # messages from its neighbors according to Ni
        # defined via the operator a
        embeddings = self.aggregate(messages)

        output = self.update(embeddings)

        return output

    def message(self, x: Tensor) -> Tensor:
        return self.get_j(x)

    def aggregate(self, messages: Tensor):
        return self.agg(messages, self.index_i, 0)

    def update(self, embeddings) -> Tensor:
        return embeddings

    def get_i(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_i)

    def get_j(self, x: Tensor) -> Tensor:
        return x.index_select(-2, self.index_j)
