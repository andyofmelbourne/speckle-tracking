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

```bash
    python process/forward_sim.py -c process/forward_sim.ini hdf5/example/example.cxi
    python gui/speckle-gui.py hdf5/example/example.cxi
```
