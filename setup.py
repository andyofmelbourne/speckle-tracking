from setuptools import setup, find_packages
from setuptools.extension import Extension

extensions = []

setup(
    name                 = "speckle-tracking",
    version              = "2020.0",
    packages             = find_packages(),
    install_requires     = ['pyqtgraph', 'h5py', 'scipy', 'numpy', 'tqdm'],
    ext_modules          = extensions
    )
