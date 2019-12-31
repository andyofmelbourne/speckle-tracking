#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import h5py

if __name__ == '__main__':
    # get command line args and config
    sc  = 'h5_operations'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.make_mask.__doc__.split('\n\n')[0]
    
    args, params = cmdline_parser.parse_cmdline_args(sc, des)
    params = params['h5_operations']

    f = h5py.File(args.filename)
    if params['operation'] == 'cp' :
        print('cp', params['from'], '-->', params['to']) 
        f.copy(params['from'], params['to'])
    
    elif params['operation'] == 'mv' :
        print('mv', params['from'], '-->', params['to']) 
        f.move(params['from'], params['to'])
    
    elif params['operation'] == 'rm' :
        print('rm:', params['from']) 
        del f[params['from']]

    f.close()
