"""Test the scatter module."""

import torch

from topomodelx.utils.scatter import scatter


class TestScatter:
  """Test the scatter module."""
    
    def test_scatter(self):
      """Test the scatter function."""
    
        tests = [
            {
                'src': [1., 3., 2., 4., 5., 6.],
                'index': [0, 1, 0, 1, 1, 3],
                'sum': [3., 12., 0., 6.],
                'add': [3., 12., 0., 6.],
                'mean': [1.5, 4., 0., 6.],
            },
            {
                'src': [1.,1.,2., 2.],
                'index': [0, 1, 0, 1],
                'sum': [3.,3.],
                'add': [3.,3.],
                'mean': [1.5,1.5],
            },
            {
                'src': [1.,2.,1.,2.,1.,2.,1.,2.],
                'index': [0,0,0,0,1,1,2,2],
                'sum': [6.,3.,3.],
                'add': [6.,3.,3.],
                'mean': [ 1.5 , 1.5, 1.5 ],
            }
        ]
        for scat in ["add","sum","mean"]:
            sc = scatter(scat)
            for i in range(0,len(tests)):
                computed=sc(torch.tensor(tests[i]['src']),
                       torch.tensor(tests[i]['index']),
                       dim=0)
                assert (torch.all(computed == torch.tensor(tests[i][scat])))