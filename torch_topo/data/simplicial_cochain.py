"""
@author: Mustafa Hajij
"""

__all__ = ["SimplicialCochain", "SimplicialData"]


from stnets.cochain.cochain import Cochain
from stnets.topology import SimplicialComplex
from torch import Tensor


class SimplicialCochain(Cochain):
    """
    For an input simplicial complex of dimension n,
    this class stores a k-cochain ( 0 <= k <= n ) for
    some k-subcomplex on the input simplicial complex.
    It also supports saving an aux tensor which can be used
    for supervised learning on the simplicial complex.
    """

    def __init__(self, simplicial_complex, aux_tensor=None):
        if not isinstance(simplicial_complex, SimplicialComplex):
            raise TypeError(
                f"Input must be a `stnets.topology.SimplicialComplex`, got {type(simplicial_complex)}."
            )

        self.domain = simplicial_complex
        Cochain.__init__(self)
        self._num_simplices = [
            len(self.simplicial_complex.dic_order_faces(i))
            for i in range(0, self.simplicial_complex.maxdim + 1)
        ]
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

        if tensor_dimension not in self._num_simplices:
            raise ValueError(
                "Input tensor must be supported on a subcomplex in the simplicial "
                + f"complex domain, the input tensor is supported on {tensor_dimension} subcomplex, "
                + f"and existing subcomplexes support dimensions are : {self.__num_simplices}."
            )

        self._tensor = tensor
        self.supp_dimension = self._num_simplices.index(tensor_dimension)

    @property
    def aux_tensor(self):
        self._aux_tensor

    @aux_tensor.setter
    def aux_tensor(self, tensor):
        assert isinstance(tensor, Tensor)
        self._aux_tensor = tensor


class SimplicialData:
    """
    For an input simplicial complex of dimension n,
    this class stores a k-cochain ( 0 <= k <= n ) for
    every k-subcomplex on the input simplicial complex.
    It also supports saving an aux tensor which can be used
    for supervised learning on the simplicial complex.
    """

    def __init__(self, simplicial_complex, aux_tensor=None):
        if not isinstance(simplicial_complex, SimplicialComplex):
            raise TypeError(
                f"Input must be a `stnets.topology.SimplicialComplex`, got {type(simplicial_complex)}."
            )

        self.domain = simplicial_complex

        self.__num_simplices = [
            len(self.simplicial_complex.dic_order_faces(i))
            for i in range(0, self.simplicial_complex.maxdim + 1)
        ]
        self._cochains = {}
        for i in range(0, self.simplicial_complex.maxdim + 1):
            self._cochains[i] = None

        self._aux_tensor = aux_tensor

    @property
    def get_tensor(self, dim):
        return self._cochain[dim]

    def set_tensor(self, dim, tensor):
        if not isinstance(tensor, Tensor):
            raise TypeError(f"Input must be a `torch.Tensor`, got {type(tensor)}.")
        if len(tensor.shape) != 2 and len(tensor.shape) != 3:
            raise ValueError(f"Input must be a 2D or 3D tensor, got {tensor.shape}.")

        if len(tensor.shape) == 3:
            tensor_dimension = tensor.size(1)
        elif len(tensor.shape) == 2:
            tensor_dimension = tensor.size(-1)

        if tensor_dimension != self.__num_simplices[dim]:
            raise ValueError(
                "Input tensor must be supported on the "
                + f"{self.__num_simplices[dim]} subcomplex in the input simplicial "
                + f"but the input tensor is supported on the {tensor_dimension} subcomplex"
            )

        self._cochains[dim] = tensor

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
