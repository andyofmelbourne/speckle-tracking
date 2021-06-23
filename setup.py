from setuptools import setup, find_packages
import glob

setup(
    name                 = "speckle-tracking",
    version              = "2019.1",
    packages             = find_packages(),
    package_data         = {'speckle_tracking': ['*.cl', 'bin/*.ini', '*.pyx', '*.c']},
    scripts              = glob.glob('speckle_tracking/bin/*.py')
    )
