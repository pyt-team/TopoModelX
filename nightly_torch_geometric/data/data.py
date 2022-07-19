from typing import Any

import torch_geometric
from torch_sparse import SparseTensor


class ShadowDotDict(dict):
    def __init__(self, dict_to_shadow):
        super().__init__(dict_to_shadow)
        self.__shadow = dict_to_shadow

    def __getattr__(self, key: str):
        return self.maybe_shadow(self.__getitem__(key))

    def __setattr__(self, key, value):
        super().__setattr__(key, value)
        if key != "_ShadowDotDict__shadow":
            print("shadow set")
            self.__shadow.__setitem__(key, value)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        self.__shadow.__setitem__(key, value)

    @classmethod
    def maybe_shadow(cls, value):
        if isinstance(value, dict):
            return cls(value)
        else:
            return value


class Data(torch_geometric.data.data.Data):
    def __cat_dim__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if isinstance(value, SparseTensor) and ("_no_diag_cat" not in key):
            return (0, 1)
        elif "index" in key or "face" in key:
            return -1
        else:
            return 0

    def __inc__(self, key: str, value: Any, *args, **kwargs) -> Any:
        if "batch" in key:
            return int(value.max()) + 1
        elif "index" in key or "face" in key:
            return self.num_nodes
        else:
            return 0

    def __getattr__(self, key: str):
        return ShadowDotDict.maybe_shadow(super().__getattr__(key))
