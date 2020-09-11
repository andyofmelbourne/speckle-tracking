#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np


def main(overide={}):
    # get command line args and config
    sc  = 'calculate_phase'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.calculate_phase.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params[sc]

    # overide with input params (if any)
    params.update(overide)
    
    phase, angles, res = st.calculate_phase(
            params['pixel_map'],
            params['weight'], 
            params['wavelength'], 
            params['z'], 
            params['x_pixel_size'], 
            params['y_pixel_size'], 
            params['dss'], 
            params['dfs'], 
            False,
            params['maxiter'])
    
    out = {'phase': phase, 'angles': angles}
    out.update(res)
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/phase', file=f)



if __name__ == '__main__':
    main()
