# -*- coding: utf-8 -*-
"""
@author: Mustafa Hajij
"""

__all__ = ["Cochain"]

from torch import Tensor


class Cochain:
    def __init__(self, domain=None, tensor=None, dimension=None):
        self._domain = domain
        self._tensor = tensor
        self._support_dimension = dimension

    @property
    def domain(self):
        return self._domain

    @domain.setter
    def domain(self, value):
        self._domain = value

    @property
    def tensor(self):
        return self.tensor

    @tensor.setter
    def tensor(self, value):
        assert isinstance(value, Tensor)
        self._tensor = value

    @property
    def supp_dimension(self):
        return self._support_dimension

    @supp_dimension.setter
    def supp_dimension(self, value):
        self._support_dimension = value

    def __add__(self, other):
        assert self.dimension == other.dimension and self.domain == other.domain
        f = Cochain(domain=self.domain, tensor=self.tensor, dimension=self.dimension)
        f.tensor = self.tensor + other.tensor
        return f

    def __sub__(self, other):
        assert self.dimension == other.dimension and self.domain == other.domain
        f = Cochain(self.domain, self.tensor, self.dimension)
        f.tensor = self.tensor - other.tensor
        return f

    def __getitem__(self, key):
        return self.tensor[key]
