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
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/energy Dataset {SCALAR}
    /entry_1/sample_3        Group
    /entry_1/sample_3/geometry Group
    /entry_1/sample_3/geometry/translation Dataset {100, 3}

This tries to follow the cxi standard layed out `here <http://www.cxidb.org/cxi.html>`_.

Now the sample positions */entry_1/sample_3/geometry/translation* 
are usually provisional and will improve over time, but I think 
that all scripts in this software should output their results into 
their own parts of the cxi file, e.g::



This way, starting again is easy.





So far the code is segmented as follows:

Code organisation
=================
**speckle-tracking**
**gui**
    **widgets.py**
    collection of widgets to interface with process.py scripts
    and to generally display stuff in the cxi file
    
    **main gui**

