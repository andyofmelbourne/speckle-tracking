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
That's right, suck it up, no micro-meters for that motor and nano 
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
  python scripts that are designed to be called from the command line,
  take an .ini file and a .cxi as input, do stuff, then put the output 
  into the same (or different) .cxi file. These process scripts can 
  then be called from the gui.

utils/
  python files containing functions of general interest in projection
  imaging. 

gui/
  **widgets.py**
  collection of widgets to interface with process.py scripts
  and to generally display stuff in the cxi file
  
  **main gui**
  the main gui that a user can use to look at cxi files and
  run scripts from

 
Repetetive stuff to automate
============================
First on my list is making a gui that:
  - takes parameters then writes to an ini file
  - calls the process.py script (via the command line)
  - then displays some output 

To this end I have a wrapper qt widget at ``utils/gui/widgets/auto_build_widget.py``. 
You call it with ``proc_widget = Auto_build_widget(script_name, h5_fnam)``, 
now you have a config editor+writer on the left, a script status thing on 
the bottem and a display area, if the script outputs to standard output
something like::
    
    display: script_name/image

then the display widget will look for a dataset called ``script_name/image`` 
in the h5 file and display it in "real time". 

Second is parsing command line arguments, which is the same for every script,
e.g.::

    $python process/forward_sim.py -h
    usage: forward_sim.py [-h] [-c CONFIG] filename

    generate an example cxi file for testing

    positional arguments:
      filename              file name of the *.cxi file

    optional arguments:
      -h, --help            show this help message and exit
      -c CONFIG, --config CONFIG
                            file name of the configuration file

For this I have ``args, params = parse_cmdline_args(script_name, description)``
in ``utils/cmdline_parser.py``. If called without the -c option it looks
for a file called ``script_name.ini`` in the same directory as the cxi file,
then for the same script name in the ``process`` directory. If it finds the 
file in the process directory then it is copied into the cxi directory. 

Third is:
  - getting the params from the ini file
  - then extracting the relevant data from the cxi file
  - then converting physical parameters into "pixel space",
  - then writing all the results back into the cxi file.  
    
I have somewhat generic "get params from hdf5 files" script but it is too 
general to be very useful here. Here is the logic:

  - get the command line arguments 
  - get the config parameters
  - extract config parameters that look like ``/h5_group/blah`` from the cxi
    file. *But* if there is an roi option then only extract data within the
    roi for datasets that have the same shape as the last two dimensions of
    ``/entry_1/data_1/data`` (a bit hacky). This is dangerous for large 
    datasets...
  - Now convert sample translations into pixel locations. 
  - When the script is finished, write all data into a single h5_group within
    the cxi file and insert roi'd data into the full frames. 

Also we should be careful with scripts that use MPI, we cant have all processes
each load the full dataset of frames, so by default I guess we split over frames.
Maybe we also want the option to split over detector pixels...

This is done in ``utils/cmdline_config_cxi_reader.py`` by ``def get_all(sn, des)``
and ``write_all(params, filename, output_dict, roi=None)``.
