"""Unit tests for the tutorials."""

import subprocess
import tempfile
from pathlib import Path

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
        subprocess.check_call(args)  # noqa: S603


paths = sorted(map(str, Path("tutorials/combinatorial").glob("*.ipynb")))


@pytest.mark.parametrize("path", paths)
def test_tutorial(path):
    """Run the test of the tutorials."""
    _exec_tutorial(path)
