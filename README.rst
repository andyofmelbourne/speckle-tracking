Installation
============
The easiest way to install speckle_tracking is through `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. 

To install just download the repo:

    git clone https://github.com/andyofmelbourne/speckle-tracking.git

Then install pocl (to enable OpenCL on CPUs), followed by the package::

    cd speckle-tracking 
    conda install -c conda-forge pocl
    pip install -e .

This last line will link the installation the current directory, so that changes to the code will take immediate effect. 

Make sure that pip is the miniconda one, and not the system version, e.g.::
   
   which pip
   /home/username/programs/miniconda3/envs/test/bin/pip

this will ensure that the dependencies are installed into current conda environment and prevent poluting the system python envoriment.

Documentation
=============
https://speckle-tracking.readthedocs.io

Quick Start
===========

Make a simulated diffraction experiment, this will output a cxi file::

    forward_sim example/example.cxi

Launch the gui::

    speckle-gui example/example.cxi


