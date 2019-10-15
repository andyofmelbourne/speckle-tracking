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
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params['pixel_map_from_data']
    
    u, res  = st.pixel_map_from_data(
                         params['data'].astype(np.float32), params['xy_pix'],
                         params['whitefield'], params['mask'],
                         search_window = params['search_window'])
    
    out = { 
            'pixel_map'   : u,
            'object_map'  : res['object_map']
          }
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
