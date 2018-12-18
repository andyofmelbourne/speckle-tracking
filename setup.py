from setuptools import setup, find_packages
from setuptools.extension import Extension
#from Cython.Build import cythonize

import glob

extensions = []
extensions.append(Extension( "speckle_tracking.feature_matching", 
                            ["speckle_tracking/feature_matching.pyx",]))
extensions.append(Extension( "speckle_tracking.poly_utils", 
                            ["speckle_tracking/poly_utils.pyx",]))

setup(
    name                 = "speckle-tracking",
    version              = "0.0.0",
    packages             = find_packages(),
    install_requires     = ['pyqt5<5.11', 'pyqtgraph', 'h5py', 'scipy'],
    scripts              = glob.glob('bin/*.py'),
    package_data         = {'':'*.ini'},
    include_package_data = True,    
    entry_points         ={
                          'gui_scripts': ['speckle_gui = speckle_gui.speckle_gui:main',],
                          'console_scripts': [
                                       'fit_defocus_thon = bin.fit_defocus_thon:main',]
                          },
    ext_modules          = extensions
    )
