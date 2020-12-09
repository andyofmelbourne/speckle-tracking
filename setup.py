from setuptools import setup, find_packages
from distutils.core import Extension
import glob
import sys
import os
import numpy

try:
      from Cython.Build import cythonize
except ImportError:
      USE_CYTHON = False
else:
      USE_CYTHON = True

ext = '.pyx' if USE_CYTHON else '.c'
extension_args = {'language': 'c',
                  'extra_compile_args': ['-fopenmp'],
                  'extra_link_args': ['-fopenmp', '-Wl,-rpath,/usr/local/lib'],
                  'library_dirs': ['/usr/local/lib', os.path.join(sys.prefix, 'lib')],
                  'include_dirs': [numpy.get_include(), os.path.join(sys.prefix, 'include')],
                  'define_macros': [("NPY_NO_DEPRECATED_API", "NPY_1_7_API_VERSION")]}

extensions = [Extension(name='speckle_tracking.make_object_map_cy',
                        sources=['speckle_tracking/make_object_map_cy' + ext],
                        **extension_args),
              Extension(name='speckle_tracking.update_pixel_map_cy',
                        sources=['speckle_tracking/update_pixel_map_cy' + ext],
                        **extension_args),
              Extension(name='speckle_tracking.update_translations_cy',
                        sources=['speckle_tracking/update_translations_cy' + ext],
                        **extension_args),
              Extension(name='speckle_tracking.calc_error_cy',
                        sources=['speckle_tracking/calc_error_cy' + ext],
                        **extension_args)]

if USE_CYTHON:
      extensions = cythonize(extensions, annotate=False, language_level='3',
                             compiler_directives={'cdivision': True,
                                                  'boundscheck': False,
                                                  'wraparound': False,
                                                  'binding': True,
                                                  'embedsignature': True})

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
      install_requires=['Cython', 'numpy'],
      ext_modules=extensions,
      scripts=glob.glob('speckle_tracking/bin/*'))
