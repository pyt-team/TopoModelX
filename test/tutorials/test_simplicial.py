"""Unit tests for the tutorials."""

import glob
import subprocess
import tempfile

import pytest


def _exec_tutorial(path):
    """Execute a tutorial notebook."""
    with tempfile.NamedTemporaryFile(suffix=".ipynb") as tmp_file:
        args = [
            "jupyter",
            "nbconvert",
            "--to",
            "notebook",
            "--execute",
            "--ExecutePreprocessor.timeout=1000",
            "--ExecutePreprocessor.kernel_name=python3",
            "--output",
            tmp_file.name,
            path,
        ]
        subprocess.check_call(args)


paths = sorted(glob.glob("tutorials/simplicial/*.ipynb"))


@pytest.mark.parametrize("path", paths)
def test_tutorial(path):
    """Run the test of the tutorials."""
    _exec_tutorial(path)
