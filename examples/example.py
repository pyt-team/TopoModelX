import os
import sys

module_path = os.path.abspath(os.path.join(".."))
if module_path not in sys.path:
    sys.path.append(module_path)

from stnets.layers.stn_conv import LTN

model = LTN(3, 4)
