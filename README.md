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

_**Note:** TMX is still under development._

## Quick Tour for New Users

In this quick tour, we highlight the ease of creating and training a TNN model with only a few lines of code.

#Train your own TNN model

Below is a minimal example of using TopoModelX to load a simplicial complex dataset, define a simplicial attention network (SAN), and perform a forward pass:


```bash
import numpy as np
import toponetx as tnx
import torch
from topomodelx.nn.simplicial.san import SAN
from topomodelx.utils.sparse import from_sparse

# Step 1: Load the Karate Club dataset
dataset = tnx.karate_club(complex_type="simplicial")

# Step 2: Prepare Laplacians and node/edge features
laplacian_down = from_sparse(dataset.down_laplacian_matrix(rank=1))
laplacian_up = from_sparse(dataset.up_laplacian_matrix(rank=1))
incidence_0_1 = from_sparse(dataset.incidence_matrix(rank=1))

x_0 = torch.tensor(np.stack(list(dataset.get_simplex_attributes("node_feat").values())))
x_1 = torch.tensor(np.stack(list(dataset.get_simplex_attributes("edge_feat").values())))
x = x_1 + torch.sparse.mm(incidence_0_1.T, x_0)

# Step 3: Define the network
class Network(torch.nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super().__init__()
        self.base_model = SAN(in_channels, hidden_channels, n_layers=2)
        self.linear = torch.nn.Linear(hidden_channels, out_channels)

    def forward(self, x, laplacian_up, laplacian_down):
        x = self.base_model(x, laplacian_up, laplacian_down)
        return torch.sigmoid(self.linear(x))

# Step 4: Initialize the network and perform a forward pass
model = Network(in_channels=x.shape[-1], hidden_channels=16, out_channels=2)
y_hat_edge = model(x, laplacian_up=laplacian_up, laplacian_down=laplacian_down)
   ```

## 🤖 Installing TopoModelX

`TopoModelX` is available on PyPI and can be installed using `pip`.
Run the following command:

```bash
pip install topomodelx
```

Then install torch, torch-scatter, torch-sparse with or without CUDA depending on your needs.
```bash
pip install torch==2.0.1 --extra-index-url https://download.pytorch.org/whl/${CUDA}
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-2.0.1+${CUDA}.html
pip install torch-cluster -f https://data.pyg.org/whl/torch-2.0.0+${CUDA}.html
```
where `${CUDA}` should be replaced by either `cpu`, `cu102`, `cu113`, or `cu115` depending on your PyTorch installation (`torch.version.cuda`).


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

TMX is a part of TopoX, a suite of Python packages for machine learning on topological domains. If you find TMX useful please consider citing our software paper:

- Hajij et al. 2023. TopoX: a suite of Python packages for machine learning on topological domains

To learn more about the blueprint topological deep learning that topomodelx follows :

- Mustafa Hajij, Ghada Zamzmi, Theodore Papamarkou, Nina Miolane, Aldo Guzmán-Sáenz, Karthikeyan Natesan Ramamurthy, Tolga Birdal, Tamal K. Dey, Soham Mukherjee, Shreyas N. Samaga, Neal Livesay, Robin Walters, Paul Rosen, Michael T. Schaub.  
  [Topological Deep Learning: Going Beyond Graph Data](https://arxiv.org/abs/2206.00606) (arXiv) • [Topological Deep Learning: A Book](https://tdlbook.org/)

TMX topological neural networks are surveyed in:

- Papillon et al. 2023. Architectures of Topological Deep Learning: A Survey on Topological Neural Networks.

```
@misc{hajij2023topological,
      title={Topological Deep Learning: Going Beyond Graph Data},
      author={Mustafa Hajij and Ghada Zamzmi and Theodore Papamarkou and Nina Miolane and Aldo Guzmán-Sáenz and Karthikeyan Natesan Ramamurthy and Tolga Birdal and Tamal K. Dey and Soham Mukherjee and Shreyas N. Samaga and Neal Livesay and Robin Walters and Paul Rosen and Michael T. Schaub},
      year={2023},
      eprint={2206.00606},
      archivePrefix={arXiv},
      primaryClass={cs.LG}
}

@article{hajij2024topox,
  title={TopoX: a suite of Python packages for machine learning on topological domains},
  author={PYT-Team},
  journal={arXiv preprint arXiv:2402.02441},
  year={2024}
}

@article{papillon2023architectures,
  title={Architectures of Topological Deep Learning: A Survey of Message-Passing Topological Neural Networks},
  author={Papillon, Mathilde and Sanborn, Sophia and Hajij, Mustafa and Miolane, Nina},
  journal={arXiv preprint arXiv:2304.10031},
  year={2023}
}

```
## Funding

<img align="right" width="200" src="https://raw.githubusercontent.com/pyt-team/TopoNetX/main/resources/erc_logo.png">

Partially funded by the European Union (ERC, HIGH-HOPeS, 101039827). Views and opinions expressed are however those of the author(s) only and do not necessarily reflect those of the European Union or the European Research Council Executive Agency. Neither the European Union nor the granting authority can be held responsible for them.

Partially funded by the National Science Foundation (DMS-2134231, DMS-2134241).
