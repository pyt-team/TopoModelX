"""
@author: Mustafa Hajij
"""

__all__ = ["HypergraphCochain", "HypergraphData", "GraphCochain", "GraphData"]

from hypernetx.classes.hypergraph import Hypergraph
from networkx.classes.graph import Graph
from stnets.cochain.cochain import Cochain
from torch import Tensor


class HypergraphCochain(Cochain):
    """
    For an input hypergraph
    this class stores a cochain supported on the nodes or the
    hyperedges of the hypergraph. It also supports saving an
    aux tensor which can be used for supervised learning
    on the the hypergraph.
    """

    def __init__(self, graph, aux_tensor=None):
        if not isinstance(graph, (Graph, Hypergraph)):
            raise TypeError(
                "Input must be a `networkx.classes.graph.Graph` or a `hypernetx.classes.hypergraph.Hypergraph`."
            )

        Cochain.__init__(self)
        self.domain = graph
        self.n_nodes = len(self.domain.nodes)
        self.n_edges = len(self.domain.edges)

        self._aux_tensor = aux_tensor

    @Cochain.tensor.setter
    def tensor(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Input must be a `torch.Tensor`, got {type(tensor)}.")
        if len(tensor.shape) != 2 and len(tensor.shape) != 3:
            raise ValueError(f"Input must be a 2D or 3D tensor, got {tensor.shape}.")

        if len(tensor.shape) == 3:
            tensor_dimension = tensor.size(1)
        elif len(tensor.shape) == 2:
            tensor_dimension = tensor.size(-1)

        if tensor_dimension not in [self.n_nodes, self.n_edges]:
            raise ValueError(
                "Input tensor must be supported on the "
                + "nodes or the edges of the input h-graph and here the number of nodes/edges are"
                + f"{self.n_nodes,self.n_edges} "
                + f" but the input tensor is supported on a {tensor_dimension} dimensional structure."
            )

        self._tensor = tensor

    @property
    def aux_tensor(self):
        self._aux_tensor

    @aux_tensor.setter
    def aux_tensor(self, tensor):
        assert isinstance(tensor, Tensor)
        self._aux_tensor = tensor


class HypergraphData:
    """
    For an input hypergraph
    this class stores a cochain supported on the nodes and the
    hyperedges of the hypergraph. It also supports saving an
    aux tensor which can be used for supervised learning
    on the the hypergraph.
    """

    def __init__(self, graph, aux_tensor=None):
        if not isinstance(graph, (Graph, Hypergraph)):
            raise TypeError(
                "Input must be a `networkx.classes.graph.Graph` or a `hypernetx.classes.hypergraph.Hypergraph`."
            )

        self.domain = graph

        self._cochains = {}
        self._cochains["nodes"] = None
        self._cochains["edges"] = None

        self.n_nodes = len(self.domain.nodes)
        self.n_edges = len(self.domain.edges)

        self._aux_tensor = aux_tensor

    @property
    def node_tensor(self):
        return self._cochains["nodes"]

    @node_tensor.setter
    def node_tensor(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Input must be a `torch.Tensor`, got {type(tensor)}.")
        if len(tensor.shape) != 2 and len(tensor.shape) != 3:
            raise ValueError(f"Input must be a 2D or 3D tensor, got {tensor.shape}.")

        if len(tensor.shape) == 3:
            tensor_dimension = tensor.size(1)
        elif len(tensor.shape) == 2:
            tensor_dimension = tensor.size(-1)

        if tensor_dimension != self.n_nodes:
            raise ValueError(
                "Input tensor must be supported on the "
                + f"nodes but the input's nodes size is {self.n_nodes} "
                + f"but the input tensor is supported on  {tensor_dimension}."
            )

        self._cochains["nodes"] = tensor

    @property
    def edge_tensor(self):
        return self._cochains["edges"]

    @edge_tensor.setter
    def edge_tensor(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Input must be a `torch.Tensor`, got {type(tensor)}.")
        if len(tensor.shape) != 2 and len(tensor.shape) != 3:
            raise ValueError(f"Input must be a 2D or 3D tensor, got {tensor.shape}.")

        if len(tensor.shape) == 3:
            tensor_dimension = tensor.size(1)
        elif len(tensor.shape) == 2:
            tensor_dimension = tensor.size(-1)

        if tensor_dimension != self.n_edges:
            raise ValueError(
                "Input tensor must be supported on the "
                + f"edges and the input h-graph has num of edges : {self.n_edges} "
                + f"but the input tensor is supported on a {tensor_dimension} dimensional structure."
            )

        self._cochains["edges"] = tensor

    @property
    def aux_tensor(self):
        self._aux_tensor

    @aux_tensor.setter
    def aux_tensor(self, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Input must be a `torch.Tensor`, got {type(tensor)}.")
        self._aux_tensor = tensor

    def __getitem__(self, key):
        return self._cochains[key]


GraphCochain = HypergraphCochain
GraphData = HypergraphData
