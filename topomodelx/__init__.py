"""
TopoModelX: Higher Order Deep Models For Python
==================================--
stnet is a Python module integrating higher order deep learning learning.
It aims to provide simple and efficient solutions to higher order deep learning
 as a versatile tool for science and engineering.


Import main modules

from .version import version as __version__
"""
from .nn import *
from .nn.cxn import *
from .nn.conv import *
from .nn.autoencoders import *
from .nn.attention import *
from .nn.scatter import *
from .transforms import *
from .topomodelx import version as __version__
