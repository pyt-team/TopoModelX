import os
from codecs import open
from os import path

from setuptools import find_packages, setup

# Package meta-data.
NAME = "torch_topo"
DESCRIPTION = "Python module integrating higher order deep learning."
URL = "https://github.com/pyt-team/torch_topo"
VERSION = 0.1

here = path.abspath(path.dirname(__file__))

with open(path.join(here, "README.md"), encoding="utf-8") as f:
    long_description = f.read()

install_requires = [
    "decorator",
    "tqdm",
    "numpy",
    "scipy",
    "requests",
    "scikit-learn",
    "networkx",
    "gudhi",
    "torch_geometric",
]

full_requires = [
    "pandas",
    "matplotlib",
    "jupyter",
    # 'torch>=1.9.0',
    # 'pytorch-memlab',
    # 'torchmetrics>=0.7'
]

test_requires = [
    "pytest",
    "pytest-cov",
]

dev_requires = test_requires + [
    "pre-commit",
    "flake8",
    "yapf",
    "black==22.6.0",
    "black[jupyter]",
    # "flake8-docstrings",  # flake8-docstrings causes flake8 to change behaviour for tests; removed it
    "isort==5.10.1",
    "codecov",
    "coverage",
]

setup(
    name=NAME,
    version=VERSION,
    description="Python module integrating higher order deep learning learning.",
    long_description="stnets aims to provide simple and efficient solutions to higher order deep learning m as a versatile tool for science and engineering.",
    url=URL,
    download_url=URL,
    license="MIT",
    author="PyT-Team Authors",
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
    python_requires=">=3.7",
    install_requires=install_requires,
    extras_require={
        "full": full_requires,
        "test": test_requires,
        "dev": dev_requires,
    },
    packages=find_packages(),
)
