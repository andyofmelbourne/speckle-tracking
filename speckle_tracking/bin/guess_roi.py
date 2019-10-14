#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

if __name__ == '__main__':
    # get command line args and config
    sc  = 'guess_roi'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.guess_roi.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=False)
    params = params['guess_roi']
    
    roi = st.guess_roi(params['whitefield'])
    
    out = {'roi': roi}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)

