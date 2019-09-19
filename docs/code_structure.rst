Code Structure (developers)
---------------------------

One problem I have is that we can't have a bunch of python files act like a bunch of scripts and a python package simultaneously, because of the import system. Fine, so let's make a python package then. The main components are:

* speckle_tracking: a bunch of utility functions to be called from python
    * some utility functions written in cython 
    * a bunch of scripts to be called from the command line
* the gui, that relies on pyqt stuff
* the docs, that relies on sphinx

I want the user to be able to download the project and run :code:`python setup.py install --user` so setup.py needs to be at the top level. Obviously we also want the README at the top level so that the user doesn't get lost. 

Code structure::

    ├── docs
    │   ├── CFEL_diatom_tutorial.rst
    │   ├── code_notes.rst
    │   ├── code_structure.rst
    │   ├── conf.py
    │   ├── coordinate_system.rst
    │   ├── images
    │   │   ├── select_frames.png
    │   │   ├── stitch.png
    │   │   └── update_pixel_map.png
    │   ├── index.rst
    │   └── Makefile
    ├── README.md
    ├── setup.py
    ├── speckle_gui
    │   ├── __init__.py
    │   ├── speckle-gui.ini
    │   ├── speckle_gui.py
    │   ├── widgets
    │   │   ├── auto_build_widget.py
    │   │   ├── config_editor_widget.py
    │   │   ├── fit_defocus_widget.py
    │   │   ├── grad_descent_widget.py
    │   │   ├── manual_tracking_widget.py
    │   │   ├── mask_maker_widget.py
    │   │   ├── run_and_log_command.py
    │   │   ├── select_frames_widget.py
    │   │   ├── show_frames_selection_widget.py
    │   │   ├── show_h5_list_widget.py
    │   │   ├── show_nd_data_widget.py
    │   │   ├── update_pixel_map_widget.py
    │   │   └── view_h5_data_widget.py
    │   └── widgets.py
    └── speckle_tracking
        ├── __init__.py
        ├── mpiarray.py
        ├── add_distortions.py
        ├── bin
        │   ├── __init__.py
        │   ├── fit_defocus_registration.ini
        │   ├── fit_defocus_registration.py
        │   ├── fit_defocus_thon.ini
        │   ├── fit_defocus_thon.py
        │   ├── forward_sim.ini
        │   ├── forward_sim.py
        │   ├── h5_operations.ini
        │   ├── h5_operations.py
        │   ├── make_whitefield.ini
        │   ├── make_whitefield.py
        │   ├── pos_refine.ini
        │   ├── pos_refine.py
        │   ├── stitch.ini
        │   ├── stitch.py
        │   ├── update_pixel_map.ini
        │   ├── update_pixel_map.py
        │   ├── zernike.ini
        │   └── zernike.py
        ├── cmdline_config_cxi_reader.py
        ├── cmdline_parser.py
        ├── config_reader.py
        ├── cython
        │   ├── feature_matching.c
        │   ├── feature_matching.cpython-36m-x86_64-linux-gnu.so
        │   ├── feature_matching.pyx
        │   ├── poly_utils.c
        │   └── poly_utils.pyx
        ├── optics
        │   ├── fit_Zernike.py
        │   └── __init__.py
        ├── scratch
        │   ├── add_distortions.py
        │   ├── atlas_builder_gui.py
        │   ├── build_atlas.py
        │   └── temp.py
        └── setup.py
