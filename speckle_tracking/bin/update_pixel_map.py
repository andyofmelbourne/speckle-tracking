#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np

def main(overide={}):
    # get command line args and config
    sc  = 'update_pixel_map'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.update_pixel_map.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params[sc]

    # overide with input parameters (if any)
    params.update(overide)
    
    u, res = st.update_pixel_map(
            params['data'].astype(np.float32),
            params['mask'], 
            params['whitefield'], 
            params['reference_image'], 
            params['pixel_map'], 
            params['n0'], 
            params['m0'], 
            params['pixel_translations'], 
            params['search_window'],
            None, None,
            params['subpixel'], 
            params['subsample'], 
            params['interpolate'], 
            params['fill_bad_pix'],
            params['quadratic_refinement'],
            params['integrate'], 
            params['clip'], 
            params['filter'], 
            verbose=True, guess=False)
    
    u0 = np.array(np.indices(params['data'].shape[1:]))
    du = u-u0
    out = {'pixel_map': u, 'pixel_map_residual': du}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/pixel_map_residual', file=f)


if __name__ == '__main__':
    main()
