#!/usr/bin/env python
import os
import numpy as np
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

if __name__ == '__main__':
    # get command line args and config
    sc  = 'make_pixel_translations'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.make_pixel_translations.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=False)
    params = params['make_pixel_translations']
    
    M = params['z'] / params['defocus']
    dx_ref  = params['x_pixel_size'] / M
    dy_ref  = params['y_pixel_size'] / M
    xy_pix  = st.make_pixel_translations(params['translations'], params['basis'], dx_ref, dy_ref)
    
    out = { 
            'dxy_ref' : np.array([dx_ref, dy_ref]),
            'xy_pix'  : xy_pix
          }
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)

