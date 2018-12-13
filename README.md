Installation
============
To install just download a [release](https://github.com/andyofmelbourne/ptychography-workspace/releases). 
Or clone the repo:

```bash
    git clone https://github.com/andyofmelbourne/speckle-tracking.git
```

Then compile the cython code:

```bash
    cd utils 
    python setup.py build_ext --inplace
    cd ..
```


quick start
===========

```bash
    python process/forward_sim.py -c process/forward_sim.ini hdf5/example/example.cxi

    python gui/speckle-gui.py hdf5/example/example.cxi
```
