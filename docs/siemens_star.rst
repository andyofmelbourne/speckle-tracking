.. _siemens_star:

Siemens Star
============

First get the siemens_star cxi file (link to come). 

For a more detailed introduction to the speckle-tracking interface, start with the :ref:`diatom` tutorial. 

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
    /entry_1/sample_1        Group
    /entry_1/sample_1/geometry Group
    /entry_1/sample_1/geometry/translation Dataset {400, 3}


This is the minimal amount of information that the input cxi file can have, see :ref:`cxi-file`. So, as we can see in the :code:`entry_1/data_1/data` the dataset consists of 400 frames, where each frame is an image of 480x438 pixels.

For this tutorial, we will show how to reconstruct the wavefield and sample reference image using the `Python Interface`_ and the `Command-line Interface`_. If you are using the GUI interface, then follow the `Command-line Interface`_ with the corresponding widget and input values.

Python Interface
----------------

Make the mask
    First let's import speckle tracking and things, then call the :py:func:`~speckle_tracking.make_mask` function with default settings to create a binary True/False (good/bad) pixel map for the detector. Then we are going to write this back into the file::

        import speckle_tracking as st
        import h5py
        import numpy
        
        # extract data
        f = h5py.File('siemens_star.cxi', 'r')

        data  = f['/entry_1/data_1/data'][()].astype(np.float32)
        basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
        z     = f['/entry_1/instrument_1/detector_1/distance'][()]
        x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
        y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
        wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
        translations = f['/entry_1/sample_1/geometry/translation'][()]
        
        f.close()
        
        mask  = st.make_mask(data)
        

Generate the Whitefield
    Now we make the "whitefield" which is what I call the image formed on the detector when there is no sample in place. You might already have this from a separate measurement, but usually it's better to estimate it directly from the scan data which we do by calling :py:func:`~speckle_tracking.make_whitefield`::

        W = st.make_whitefield(data, mask)
        
Define the ROI 
    Usually the region of the detector with useful diffraction is small compared to the full detector area. So defining the ROI (Region Of Interest) speeds things up, do this manually or by using a script that tries to guess this region :py:func:`~speckle_tracking.guess_roi`::
        
        roi = st.guess_roi(W)
        
Determine the defocus
    Now let's refine the focus to sample distance :py:func:`~speckle_tracking.fit_defocus`:: 
        
        defocus, res = st.fit_defocus(
                          data,
                          x_pixel_size, y_pixel_size,
                          z, wav, mask, W, roi)
        
Generate the pixel mapping
    Now let us estimate the geometric distortions of each image from the defocus 
    using :py:func:`~speckle_tracking.make_pixel_map`, and the astigmatism (dz)::
        
        pixel_map, pixel_map_inv, dxy = st.make_pixel_map(
                                           z, defocus, dz, roi, 
                                           x_pixel_size, y_pixel_size, 
                                           W.shape)
    
Form the object image
    Now we make a projection image of the sample using 
    :py:func:`~speckle_tracking.make_pixel_translations` and :py:func:`~speckle_tracking.make_object_map`, 
    which will be somewhat blurry because of the lens aberrations::
        
        dij_n = st.make_pixel_translations(translations, basis, dxy[0], dxy[1])
        
        O, n0, m0 = st.make_object_map(data, mask, W, dij_n, pixel_map)

Determine the lens pupil function
    Now that we have an estimate of the object projection image, we can refine the 
    :py:func:`~speckle_tracking.update_pixel_map` which can then be used to form the pupil function::
        
        pixel_map, res = st.update_pixel_map(
                            data, mask, W, O, pixel_map, 
                            n0, m0, dij_n, search_window=20)

Refinement
    Now we have the pixel map and the object map, we can refine our estimate for all parameters 
    in the system. Here is the full working example with a basic refinement loop::
        
        import speckle_tracking as st
        import h5py
        import numpy as np

        #---------------------------
        # Read data
        #---------------------------
        f = h5py.File('siemens_star.cxi', 'r')

        data  = f['/entry_1/data_1/data'][()].astype(np.float32)
        basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
        z     = f['/entry_1/instrument_1/detector_1/distance'][()]
        x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
        y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
        wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
        translations = f['/entry_1/sample_1/geometry/translation'][()]

        f.close()


        #---------------------------
        # Intialise
        #---------------------------
        mask  = st.make_mask(data)

        W = st.make_whitefield(data, mask)

        roi = st.guess_roi(W)

        dz, res = st.fit_defocus(
                     data,
                     x_pixel_size, y_pixel_size,
                     z, wav, mask, W, roi)

        pixel_map, uinv, dxy = st.make_pixel_map(
                          z, dz, res['astigmatism'], 
                          roi, x_pixel_size, y_pixel_size,
                          W.shape)

        dij_n = st.make_pixel_translations(
                   translations, 
                   basis, dxy[0], dxy[1])

        O, n0, m0 = st.make_object_map(
                       data, mask, W, dij_n, pixel_map)

        #---------------------------
        # Refine
        #---------------------------
        for i in range(10):
            pixel_map, res = st.update_pixel_map(
                        data, mask, W, O, pixel_map, n0, m0, dij_n, 
                        clip = [-40, 40],
                        fill_bad_pix = True, 
                        integrate = True, 
                        quadratic_refinement = True)
            
            O, n0, m0  = st.make_object_map(data, mask, W, dij_n, pixel_map, subpixel=True)
            dij_n, res = st.update_translations(data, mask, W, O, pixel_map, n0, m0, dij_n)

        phase, angles, res = st.integrate_pixel_map(
                                pixel_map, W, wav, z-z1, z, 
                                x_pixel_size, y_pixel_size, 
                                dxy[0], dxy[1], False, maxiter=5000)

        propx, propy, dx, dy, dz = st.propagation_profile( 
                                      phase, W, z, wav, x_pixel_size, 
                                      y_pixel_size, 1, 
                                      zs=[-1e-4, 1e-4, 1000], Nint=4)

        #---------------------------
        # Write results
        #---------------------------
        st.write_h5({
            'object_map': O, 
            'object_map_voxel_size': dxy, 
            'n0': n0, 'm0': m0, 
            'pixel_map': pixel_map, 
            'pixel_translations': dij_n,
            'propagation_profile_ss': propx, 
            'propagation_profile_fs': propy, 
            'propagation_profile_voxel_size': np.array([dx, dy, dz]),
            'phase' : phase,
            'angles' : angles,
            'angles_forward' : res['angles_forward']
            }, og='speckle_tracking/')

.. raw:: html

    <script src="https://asciinema.org/a/14.js" id="asciicast-14" async></script>


Command-line Interface
----------------------

:: 
    
    # build the pixel mask (with default settings)
    make_mask.py siemens_star.cxi

    # build the white-field array (with default settings)
    make_whitefield.py siemens_star.cxi
    
    # estimate the significant region of interest 
    guess_roi.py siemens_star.cxi

    # estimate the defocus values by ``Thon ring'' fitting
    fit_thon_rings.py siemens_star.cxi

    # check the result of the above procedure
    hdf_display.py siemens_star.cxi/speckle_tracking/thon_display

Now we will generate the pixel map array. Because we have used the Thon rings to fit the defocus value, we also have an estimate for the degree of astigmatism. At the moment there are two defocus values stored in the cxi file::

    /speckle_tracking/defocus    (
    /speckle_tracking/defocus_fs 
    /speckle_tracking/defocus_ss 
    

    generate_pixel_map.py siemens_star.cxi
    make_reference.py siemens_star.cxi
