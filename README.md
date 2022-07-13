[![Test](https://github.com/pyt-team/torch_topo/actions/workflows/test.yml/badge.svg)](https://github.com/mhajij/stnets/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/torch_topo/actions/workflows/lint.yml/badge.svg)](https://github.com/mhajij/stnets/actions/workflows/lint.yml)

# Pytorch Topological: Higher Order Deep Models For Python

Pytrorch Topological is a Python module integrating higher order deep learning learning.
It aims to provide simple and efficient solutions to higher order deep learning
 as a versatile tool for science and engineering.

# Contributing to PyT
*based off PyGeometric*

## Developing PyTopo

To develop PyTopo on your machine, here are some tips:

1. Clone a copy of PyTopo from source:

   ```bash
   git clone https://github.com/pyt-team/torch_topo
   cd torch_topo
   ```

2. If you already cloned PyT from source, update it:

   ```bash
   git pull
   ```

3. Install PyT in editable mode:

   ```bash
   pip install -e ".[dev,full]"
   ```

   This mode will symlink the Python files from the current local source tree into the Python install. Hence, if you modify a Python file, you do not need to reinstall PyT again and again.

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      To install the binaries for PyTorch 1.12.0, simply run:

      ```
      pip install torch --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse torch-geometric -f https://data.pyg.org/whl/torch-1.12.0+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

      For PyTorch 10 or 11 replace `torch-1.12` by `torch-1.10` or `torch-1.11`.




5. Ensure that you have a working PyT installation by running the entire test suite with

   ```bash
   pytest
   ```

   In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

   ```bash
    pre-commit install
   ```

## Unit Testing

The PyT testing suite is located under `test/`.
Run the entire test suite with

```bash
pytest
```

## Black

