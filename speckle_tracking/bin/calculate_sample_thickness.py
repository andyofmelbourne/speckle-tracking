#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np

def main():
    # get command line args and config
    sc  = 'calculate_sample_thickness'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.calculate_sample_thickness.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=False)
    params = params[sc]
    
    t_pag, t_ctf = st.calculate_sample_thickness(
            params['delta'],
            params['beta'], 
            params['distance'], 
            params['defocus'], 
            params['wavelength'], 
            params['dss'], 
            params['dfs'], 
            params['reference_image'], 
            params['reference_roi'],
            params['set_median_to_zero'], 
            params['tol_ctf'], 
            params['tol_tie'])
    
    out = {'sample_thickness_pag': t_pag, 'sample_thickness_ctf': t_ctf}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/sample_thickness_ctf', file=f)



if __name__ == '__main__':
    main()
