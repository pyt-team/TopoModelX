"""Unit tests for the examples."""

import subprocess
import tempfile


def _exec_notebook(path):

    file_name = tempfile.NamedTemporaryFile(suffix=".ipynb").name
    args = [
        "jupyter",
        "nbconvert",
        "--to",
        "notebook",
        "--execute",
        "--ExecutePreprocessor.timeout=1000",
        "--ExecutePreprocessor.kernel_name=python",
        "--output",
        file_name,
        path,
    ]
    subprocess.check_call(args)


def test_introduction_to_simplicial_attention_networks():
    _exec_notebook("examples/Introduction_to_HigherOrder_Attention_Networks.ipynb")


def test_introduction_to_deep_higher_order_networks():
    _exec_notebook("examples/Introduction_to_deep_higher_order_networks.ipynb")


def test_training_higher_order_deep_networks():
    _exec_notebook("examples/Training_higher_order_deep_networks.ipynb")
