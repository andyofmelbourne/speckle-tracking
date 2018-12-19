Siemens Star
============

First get the siemens_star cxi file (link to come).

The Input CXI File
------------------
The file has the following structure::

     └─ $  h5ls -r siemens_star.cxi 
    /                        Group
    /entry_1                 Group
    /entry_1/data_1          Group
    /entry_1/data_1/data     Dataset {400, 480, 438}
    /entry_1/instrument_1    Group
    /entry_1/instrument_1/detector_1 Group
    /entry_1/instrument_1/detector_1/basis_vectors Dataset {400, 2, 3}
    /entry_1/instrument_1/detector_1/distance Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/mask Dataset {480, 438}
    /entry_1/instrument_1/detector_1/x_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/detector_1/y_pixel_size Dataset {SCALAR}
    /entry_1/instrument_1/source_1 Group
    /entry_1/instrument_1/source_1/energy Dataset {SCALAR}
    /entry_1/instrument_1/source_1/wavelength Dataset {SCALAR}
    /entry_1/sample_3        Group
    /entry_1/sample_3/geometry Group
    /entry_1/sample_3/geometry/translation Dataset {400, 3}


This is the minimal amount of information that the input cxi file can have, see :ref:`cxi-file`. So, as we can see in the :code:`entry_1/data_1/data` the dataset consists of 400 frames, where each frame is an image of 480x438 pixels.


Now that's out of the way, we should decide if we want to use the `Python Interface`_, `Command-line Interface`_ or the `Gui Interface`_. So... choose. 

Python Interface
----------------

Make the mask
    First let's import speckle tracking and things, then call the :code:`make_mask` function with default settings to create a binary True/False (good/bad) pixel map for the detector. Then we are going to write this back into the file::

        import speckle_tracking as st
        import h5py
        import numpy
        import pyqtgraph as pg
        
        # extract data
        f = h5py.File('siemens_star.cxi', 'r')

        data  = f['/entry_1/data_1/data'][()]
        basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
        z     = f['/entry_1/instrument_1/detector_1/distance'][()]
        x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
        y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
        wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
        translations = f['/entry_1/sample_3/geometry/translation'][()]
        
        f.close()
        
        mask  = st.make_mask(data)
        
        # check the result
        pg.show(mask)
        pg.show(data[0])

Determine the defocus
    Now let's refine the focus to sample distance as well as the low order aberrations. Then we make a pixel map that encodes the geometric distortions:: 
        
        defocus, aberrations = st.fit_defocus(
                                  data, mask,
                                  x_pixel_size, y_pixel_size,
                                  distance, 
                                  wav, basis,
                                  translations
                                  )
        
        pixel_shifts = st.pixel_shifts_aberrations(
                                  aberrations, 
                                  x_pixel_size, y_pixel_size,
                                  distance, 
                                  wav, basis
                                  )
       
    
Form the object image
    Now we make a projection image of the sample, which will be somewhat blurry because of the lens aberrations::
        
        O, coords = st.stitch(data, mask,
                              pixel_shifts)

Determine the lens pupil function
    Now that we have an estimate of the object projection image, we can refine the :code:`pixel_shifts` which can then be used to form the pupil function::
        
        phase, pixel_shifts = st.pixel_shifts_data(data, mask,
                                                   pixel_shifts)

Command-line Interface
----------------------

Gui Interface
-------------
