#!/usr/bin/env python
import os
import numpy as np
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

# HACK add bin to path
#---------------------
# this means that we can call the 'main()' function of each process
# so that we do not have to spawn additional child processes 
# that (for some reason) I cannot kill with the GUI
import sys
sys.path.append(os.path.join(os.path.dirname(st.__file__), 'bin'))

if __name__ == '__main__':
    # get command line args and config
    sc  = 'pixel_map_from_data'
    
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.pixel_map_from_data.__doc__.split('\n\n')[0]
    
    # get command line args and config params
    args, params = cmdline_parser.parse_cmdline_args(sc, des, config_dirs=config_dirs)
    params = params['pixel_map_from_data']
    
    # Initialisation
    #---------------
    # this stops the cmd line arguent "-c pixel_map_from_data.ini" from
    # overiding the default configs of the commands below
    sys.argv = sys.argv[:2]

    # make mask
    if params['mask'] == 'auto':
        import make_mask
        make_mask.main()

    # make whitefield
    if params['whitefield'] == 'auto':
        import make_whitefield
        make_whitefield.main()
                       
    # make ROI
    if params['roi'] == 'auto':
        import guess_roi
        guess_roi.main()
    
    # defocus
    if params['defocus'] == 'auto':
        import fit_thon_rings
        fit_thon_rings.main()

    # pixel map
    if params['pixel_map'] == 'auto':
        import generate_pixel_map
        generate_pixel_map.main()
     
    # Main Loop
    #----------
    for i in range(3):
        # make reference
        import make_reference
        make_reference.main()
        
        # update pixel map
        import update_pixel_map
        update_pixel_map.main()
        
        # update translations
        import update_translations
        update_translations.main()
        
        # calculate error
        import calc_error
        calc_error.main()
    
    # Adiational analysis
    #--------------------
    # calculate sample thickness
    import calculate_sample_thickness
    calculate_sample_thickness.main()
    
    # calculate phase
    import calculate_phase
    calculate_sample_phase.main()

    # calculate focus profile
    import focus_profile
    focus_profile.main()

