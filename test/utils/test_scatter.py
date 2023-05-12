"""Test the scatter module."""

import torch

from topomodelx.utils.scatter import scatter


class TestScatter:
    """Test the scatter module."""

    def test_scatter(self):
        """Test the scatter function."""
        tests = [
            {
                "src": [1.0, 3.0, 2.0, 4.0, 5.0, 6.0],
                "index": [0, 1, 0, 1, 1, 3],
                "sum": [3.0, 12.0, 0.0, 6.0],
                "add": [3.0, 12.0, 0.0, 6.0],
                "mean": [1.5, 4.0, 0.0, 6.0],
            },
            {
                "src": [1.0, 1.0, 2.0, 2.0],
                "index": [0, 1, 0, 1],
                "sum": [3.0, 3.0],
                "add": [3.0, 3.0],
                "mean": [1.5, 1.5],
            },
            {
                "src": [1.0, 2.0, 1.0, 2.0, 1.0, 2.0, 1.0, 2.0],
                "index": [0, 0, 0, 0, 1, 1, 2, 2],
                "sum": [6.0, 3.0, 3.0],
                "add": [6.0, 3.0, 3.0],
                "mean": [1.5, 1.5, 1.5],
            },
        ]
        for scat in ["add", "sum", "mean"]:
            sc = scatter(scat)
            for i in range(0, len(tests)):
                computed = sc(
                    torch.tensor(tests[i]["src"]),
                    torch.tensor(tests[i]["index"]),
                    dim=0,
                )
                assert torch.all(computed == torch.tensor(tests[i][scat]))
