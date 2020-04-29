from setuptools import setup, find_packages
import glob

setup(
    name                 = "speckle-tracking",
    version              = "2019.1",
    packages             = find_packages(),
    scripts              = glob.glob('speckle_tracking/bin/*.py')
    )
