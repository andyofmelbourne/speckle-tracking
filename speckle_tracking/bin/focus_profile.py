#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np

def main(overide={}):
    # get command line args and config
    sc  = 'focus_profile'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.focus_profile.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params[sc]

    # overide with input params (if any)
    params.update(overide)
    
    px, py, dx, dy, zstep = st.focus_profile(
            params['phase'],
            params['whitefield'], 
            params['z'], 
            params['wavelength'], 
            params['x_pixel_size'], 
            params['y_pixel_size'], 
            params['zs'], 
            params['subsamples'])
    
    out = {'profile_ss': px, 'profile_fs': py, 'xyz_voxel_size': np.array([dx, dy, zstep])}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/profile_ss', file=f)



if __name__ == '__main__':
    main()
