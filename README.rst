Installation (Linux)
====================
The easiest way to install speckle_tracking is through `Miniconda <https://docs.conda.io/en/latest/miniconda.html>`_. 

To install just download the repo::

    git clone https://github.com/andyofmelbourne/speckle-tracking.git

Then install pocl (to enable OpenCL on CPUs) and pyopencl, followed by the package::

    cd speckle-tracking 
    conda install -c conda-forge pocl pyopencl pyqt
    pip install -e .

This last line will link the installation the current directory, so that changes to the code will take immediate effect. 
You may also want to add the speckle_tracking functions to your path::

    echo PATH=`pwd`/speckle_tracking/bin:$PATH >> ~/.bashrc

Note on pip
    Make sure that pip is the miniconda one, and not the system version, e.g.::

        which pip
        >> /home/username/programs/miniconda3/envs/test/bin/pip

    this will ensure that the dependencies are installed into the current conda environment and prevent polluting the system python environment.


Documentation
=============
https://speckle-tracking.readthedocs.io

Getting Started
===============
Start with the `Diatom tutorial <https://speckle-tracking.readthedocs.io/en/latest/CFEL_diatom_tutorial.html>`_, then do the `Simens star tutorial <https://speckle-tracking.readthedocs.io/en/latest/siemens_star.html>`_

