from setuptools import setup, find_packages
from setuptools.extension import Extension
#from Cython.Build import cythonize

import glob

extensions = []
#extensions.append(Extension( "speckle_tracking.feature_matching", 
#                            ["speckle_tracking/feature_matching.pyx",]))
#extensions.append(Extension( "speckle_tracking.poly_utils", 
#                            ["speckle_tracking/poly_utils.pyx",]))

setup(
    name                 = "speckle-tracking",
    version              = "2019.0",
    packages             = find_packages(),
    install_requires     = ['pyqt5', 'pyqtgraph', 'h5py', 'scipy', 'numpy', 'tqdm'],
    ext_modules          = extensions
    )
