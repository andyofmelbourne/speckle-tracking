from setuptools import setup, find_packages
setup(
    name="speckle-tracking",
    version="0.0.0",
    packages=find_packages(),
    install_requires=['pyqtgraph', 'h5py', 'scipy'],
    scripts=['bin/forward_sim.py'],
    package_data = {'':'*.ini'},
    include_package_data=True,    
)
