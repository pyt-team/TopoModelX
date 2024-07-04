<h2 align="center">
  <img src="https://raw.githubusercontent.com/pyt-team/TopoModelX/main/resources/logo.png" height="250">
</h2>

<h3 align="center">
    Building Topological Neural Networks for Topological Deep Learning
</h3>

<p align="center">
  <a href="#-contributing-to-tmx">Contributing to TMX</a> •
  <a href="#-references">References</a>
</p>

<div align="center">

[![Test Codebase](https://github.com/pyt-team/torch_topo/actions/workflows/test_codebase.yml/badge.svg)](https://github.com/pyt-team/torch_topo/actions/workflows/test_codebase.yml)
[![Test Tutorials](https://github.com/pyt-team/torch_topo/actions/workflows/test_tutorials.yml/badge.svg)](https://github.com/pyt-team/torch_topo/actions/workflows/test_tutorials.yml)
[![Lint](https://github.com/pyt-team/torch_topo/actions/workflows/lint.yml/badge.svg)](https://github.com/pyt-team/torch_topo/actions/workflows/lint.yml)
[![Codecov](https://codecov.io/gh/pyt-team/TopoModelX/branch/main/graph/badge.svg)](https://app.codecov.io/gh/pyt-team/TopoModelX)
[![Docs](https://img.shields.io/badge/docs-website-brightgreen)](https://pyt-team.github.io/topomodelx/index.html)
[![Python](https://img.shields.io/badge/python-3.10+-blue?logo=python)](https://www.python.org/)
[![license](https://badgen.net/github/license/pyt-team/TopoNetX?color=green)](https://github.com/pyt-team/TopoNetX/blob/main/LICENSE)

[![slack](https://img.shields.io/badge/chat-on%20slack-purple?logo=slack)](https://join.slack.com/t/pyt-teamworkspace/shared_invite/zt-2k63sv99s-jbFMLtwzUCc8nt3sIRWjEw)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.7958513.svg)](https://doi.org/10.5281/zenodo.7958513)

</div>



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

For example, create a conda environment:
   ```bash
   conda create -n tmx python=3.11.3
   conda activate tmx
   ```

Then:

1. Clone a copy of tmx from source:

   ```bash
   git clone git@github.com:pyt-team/TopoModelX.git
   cd TopoModelX
   ```

2. Install tmx in editable mode:

   ```bash
   pip install -e '.[all]'
   ```
   **Notes:**
   - Requires pip >= 21.3. Refer: [PEP 660](https://peps.python.org/pep-0660/).
   - On Windows, use `pip install -e .[all]` instead (without quotes around `[all]`).

4. Install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.

      ```bash
      pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
      pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
      pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
      ```

      where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).

5. Ensure that you have a working tmx installation by running the entire test suite with

   ```bash
   pytest
   ```

   In case an error occurs, please first check if all sub-packages ([`torch-scatter`](https://github.com/rusty1s/pytorch_scatter), [`torch-sparse`](https://github.com/rusty1s/pytorch_sparse), [`torch-cluster`](https://github.com/rusty1s/pytorch_cluster) and [`torch-spline-conv`](https://github.com/rusty1s/pytorch_spline_conv)) are on its latest reported version.

6. Install pre-commit hooks:

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
## Funding

<img align="right" width="200" src="https://raw.githubusercontent.com/pyt-team/TopoNetX/main/resources/erc_logo.png">

Partially funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

Partially funded by the National Science Foundation (DMS-2134231, DMS-2134241).
