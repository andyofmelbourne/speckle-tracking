#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

def main(overide={}):
    # get command line args and config
    sc  = 'make_whitefield'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.make_whitefield.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=False)
    params = params['make_whitefield']

    # overide with input parameters (if any)
    params.update(overide)
    
    W = st.make_whitefield(params['data'], params['mask'])
    
    out = {'whitefield': W}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)

    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/whitefield', file=f)

if __name__ == '__main__':
    main()
