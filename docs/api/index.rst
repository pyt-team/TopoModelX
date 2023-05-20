=============
API Reference
=============

The API reference gives an overview of `TopoModelX`, which consists of several modules:

- `classes` implements the topological domains: simplicial complexes, cellular complexes, combinatorial complexes.
- `algorithms` implements signal processing techniques on topological domains, such as the eigendecomposition of a laplacian.
- `datasets` implements utilities to load small datasets on topological domains.
- `transform` implements functions to transform the topological domain that supports a dataset, effectively "lifting" the dataset onto another domain.


.. toctree::
   :maxdepth: 2
   :caption: Packages & Modules

   datasets
   classes
   transform
   algorithms
