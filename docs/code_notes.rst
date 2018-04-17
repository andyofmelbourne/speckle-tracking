CXI files
=========
This software primarily deals with projection images, that are stored
as .cxi files. The basic cxi file contains the raw image data as well
as sample positions, beam energy and so on. It is structured as so::

    $h5ls -r hdf5/example/example.cxi 
    /                        Group
    /entry_1                 Group
    /entry_1/data_1          Group
    /entry_1/data_1/data     Dataset {100, 256, 256}
    /entry_1/instrument_1    Group
    /entry_1/instrument_1/detector_1 Group
    /entry_1/instrument_1/detector_1/basis_vectors Dataset {100, 2, 3}
    /entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/mask Dataset {256, 256}
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/energy Dataset {SCALAR}
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_3        Group
    /entry_1/sample_3/geometry Group
    /entry_1/sample_3/geometry/translation Dataset {100, 3}

This tries to follow the cxi standard, layed out 
`here <http://www.cxidb.org/cxi.html>`_. But by far the most 
important aspect of cxi files is that *everything* is in SI units.
Thats right, suck it up, no micro-meters for that motor and nano 
meters for this motor or eV for that energy or mJ's for pulse energy!

Now the sample positions */entry_1/sample_3/geometry/translation* 
are usually provisional and will improve over time, but I think 
that all scripts in this software should output their results into 
their own parts of the cxi file, e.g::

    /pos_refinement              Group
    /pos_refinement/translation  Dataset {100, 3}

where "pos_refinement" is the name of the script. This way, starting 
again is easy. 

In general I do not want to rely on the cxi standard too much, 
which will involve a lot parameters for input / output. If possible
each script should ask for the data and parameters it needs. So, in
general, a script should get it's input from all and sundry then
ouput its stuff to a h5 group as above.


Code organisation
=================
speckle-tracking

process/
  python scripts that are designed to take an .ini file and a .cxi
  as input, do stuff, then put the output into the same (or different)
  .cxi file. These process scripts can then be called from the gui.

utils/
  python files containing functions of general intereset in projection
  imaging. 

gui/
  **widgets.py**
  collection of widgets to interface with process.py scripts
  and to generally display stuff in the cxi file
  
  **main gui**
  the main gui that a user can use to look at cxi files and
  run scripts from

