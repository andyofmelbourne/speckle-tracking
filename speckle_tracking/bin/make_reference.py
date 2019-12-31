#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

if __name__ == '__main__':
    # get command line args and config
    sc  = 'make_reference'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.make_object_map.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params['make_reference']
    
    O, n0, m0 = st.make_object_map(
                           params['data'], 
                           params['mask'], 
                           params['whitefield'], 
                           params['pixel_translations'], 
                           params['pixel_map'], 
                           subpixel=True)
    
    out = {'reference_image': O, 'n0': n0, 'm0': m0}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)

    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/reference_image', file=f)
