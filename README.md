Installation
============
To install just download the repo:

```bash
    git clone https://github.com/andyofmelbourne/speckle-tracking.git
```

Then install:

```bash
    cd speckle-tracking 
    python setup.py develop
```

quick start
===========

Make a simulated diffraction experiment, this will output a cxi file:
```bash
    forward_sim.py hdf5/example/example.cxi
    python bin/speckle-gui.py hdf5/example/example.cxi
```

Launch the gui:
```bash
    python bin/speckle-gui.py hdf5/example/example.cxi
```
