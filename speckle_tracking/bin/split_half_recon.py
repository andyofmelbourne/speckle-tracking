#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np

if __name__ == '__main__':
    # get command line args and config
    sc  = 'split_half_recon'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.split_half_recon.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params[sc]

    u1, u2, res1, res2, res3 = st.split_half_recon(
            params['data'].astype(np.float32),
            params['mask'], 
            params['whitefield'], 
            params['reference_image'], 
            params['pixel_map'], 
            params['n0'], 
            params['m0'], 
            params['pixel_translations'], 
            params['dss'], 
            params['dfs'], 
            params['distance'],
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
    
    out = {'pixel_map_difference': u1-u2}
    out.update(res3)
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/pixel_map_difference', file=f)


