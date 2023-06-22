[![Test](https://github.com/pyt-team/torch_topo/actions/workflows/test.yml/badge.svg)](https://github.com/pyt-team/torch_topo/actions/workflows/test.yml)
[![Lint](https://github.com/pyt-team/torch_topo/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/torch_topo/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoModelX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoModelX)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7958513.svg)](https://doi.org/10.5281/zenodo.7958513)


![pyt](https://github.com/mhajij/shrec_16/blob/main/logo.png)


# 🌐 TopoModelX (TMX) 🍩
## Topological Deep Learning

![tnns_network_with_layers](https://user-images.githubusercontent.com/8267869/234084036-f7d6585e-b7c2-4156-a825-cfa5b9658d71.png)

`TopoModelX` (TMX) is a Python module for topological deep learning. It offers simple and efficient tools to implement topological neural networks for science and engineering.

TMX's development follows the topological deep learning (TDL) blue print laid out in:
- [Hajij et al. 2023. Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606).

TMX can reproduce and extend the topological neural networks (TNNs) surveyed in:
- [Papillon et al. 2023. Architectures of Topological Deep Learning: A Survey on Topological Neural Networks](https://arxiv.org/abs/2304.10031).

See [our graphical literature review](https://github.com/pyt-team/TopoModelX/blob/main/topomodelx.jpeg) with message-passing equations available at [https://github.com/awesome-tnns/awesome-tnns](https://github.com/awesome-tnns/awesome-tnns).

_**Note:** TMX is still under development._

## 🦾 Contributing to TMX

To develop tmx on your machine, here are some tips.

First, we recommend using Python 3.11.3, which is the python version used to run the unit-tests.

Then:

1. Clone a copy of tmx from source:

   ```bash
   git clone git@github.com:pyt-team/TopoModelX.git
   cd TopoModelX
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e .[all]
   ```

3. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

4. Ensure that you have a working tmx installation by running the entire test suite with

   ```bash
   pytest
   ```

   In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

5. Install pre-commit hooks:

   ```bash
   pre-commit install
   ```

## 🔍 References ##

To learn more about the topological deep learning blueprint:

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub. [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606).
```
@misc{hajij2023topological,
      title={Topological Deep Learning: Going Beyond Graph Data},
      author={Mustafa Hajij and Ghada Zamzmi and Theodore Papamarkou and Nina Miolane and Aldo Guzmán-Sáenz and Karthikeyan Natesan Ramamurthy and Tolga Birdal and Tamal K. Dey and Soham Mukherjee and Shreyas N. Samaga and Neal Livesay and Robin Walters and Paul Rosen and Michael T. Schaub},
      year={2023},
      eprint={2206.00606},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}
```
