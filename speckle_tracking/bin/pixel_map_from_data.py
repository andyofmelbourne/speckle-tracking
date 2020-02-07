#!/usr/bin/env python
import os
import numpy as np
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

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
    # make mask
    if params['mask'] == 'auto':
        os.system("make_mask.py " + args.filename)

    # make whitefield
    if params['whitefield'] == 'auto':
        os.system("make_whitefield.py " + args.filename)
                       
    # make ROI
    if params['roi'] == 'auto':
        os.system("guess_roi.py " + args.filename)

    # defocus
    if params['defocus'] == 'auto':
        os.system("fit_thon_rings.py " + args.filename)

    # pixel map
    if params['pixel_map'] == 'auto':
        os.system("generate_pixel_map.py " + args.filename)
     
    # Main Loop
    #----------
    for i in range(3):
        # make reference
        os.system("make_reference.py " + args.filename)
        
        # update pixel map
        os.system("update_pixel_map.py " + args.filename)
        
        # update translations
        os.system("update_translations.py " + args.filename)
        
        # calculate error
        os.system("calc_error.py " + args.filename)
    
    # Adiational analysis
    #--------------------
    # calculate sample thickness
    os.system("calculate_sample_thickness.py " + args.filename)
    
    # calculate phase
    os.system("calculate_phase.py " + args.filename)

    # calculate focus profile
    os.system("focus_profile.py " + args.filename)

