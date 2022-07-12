import os
from codecs import open
from os import path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "stnets"
DESCRIPTION = "Python module integrating higher order deep learning."
URL = "https://github.com/mhajij/stnets"
VERSION = 0.2
REQUIRED = ["numpy", "torch>=1.9.0", "scipy", "scikit-learn"]


here = path.abspath(path.dirname(__file__))


with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

setup(
    name=NAME,
    version=VERSION,
    description="Python module integrating higher order deep learning learning.",
    long_description="stnets aims to provide simple and efficient solutions to higher order deep learning m as a versatile tool for science and engineering.",
    url=URL,
    download_url=URL,
    license="MIT",
    author="Simplicial Tensor Networks Authors",
    contact_email="mustafahajij@gmail.com",
    classifiers=[
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: MIT License",
        "Natural Language :: English",
        "Programming Language :: Python :: 3",
    ],
    keywords=[
        "higher order deep learning",
        "higher order networks",
        "Higher Order Representation learning",
        "Simplicial Attention Networks",
        "Simplicial Complex Neural Networks",
        "Tensor networks" "Cell complex Neural Networks",
        "Cellular complex Neural Networks",
        "CW Neural Networks",
    ],
    python_requires=">=3.6",
    install_requires=REQUIRED,
)
