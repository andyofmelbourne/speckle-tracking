from setuptools import setup, find_packages
import glob
import sys
import os

with open('README.rst', 'r') as readme:
      long_description = readme.read()

setup(name='speckle-tracking',
      version='2020.1',
      long_description=long_description,
      long_description_content_type='text/rst',
      url='https://github.com/andyofmelbourne/speckle-tracking',
      packages=find_packages(),
      include_package_data=True,
      package_data={'speckle_tracking': ['*.cl', 'bin/*.ini', '*.pyx', '*.c']},
      install_requires=['numpy'],
      scripts=glob.glob('speckle_tracking/bin/*.py'))
