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

Change the colour scale on the Frame view until you have good contrast. Then you can drag the yellow vertical line near the bottom of the display to brows the frames. To perform a basic stitch of these frames (like a montage), select the "show stitch" tab and click "Calculate stitch". After a while you should see the retrieved object.
