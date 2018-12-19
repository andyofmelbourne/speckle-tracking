Installation
============
To install just download the repo:

    git clone https://github.com/andyofmelbourne/speckle-tracking.git

Then install::

    cd speckle-tracking 
    python setup.py install --user

Quick Start
===========

Make a simulated diffraction experiment, this will output a cxi file::

    forward_sim example/example.cxi

Launch the gui::

    speckle-gui example/example.cxi
