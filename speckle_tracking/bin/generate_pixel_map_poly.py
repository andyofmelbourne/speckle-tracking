#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

import numpy as np

def main(overide={}):
    # get command line args and config
    sc  = 'make_pixel_map_poly'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    #des = st.generate_pixel_map.__doc__.split('\n\n')[0]
    des = ""
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params[sc]
    
    # overide with input params (if any)
    params.update(overide)
    
    # evaluate the polynomial expreesion for the pixel map
    y, x = np.indices(params['whitefield'].shape)    
    ufs = eval(params['pixel_map_fs'])
    uss = eval(params['pixel_map_ss'])
    u = np.array([uss, ufs])
    u = np.clip(u, -1000, 1000)

    # generate the pixel translations
    M = params['z'] / params['defocus']
    dfs = params['x_pixel_size']/ M
    dss = params['y_pixel_size'] / M

    pixel_translations = st.make_pixel_translations(
            params['translations'], 
            params['basis'], 
            dss, dfs)

    O, n0, m0 = st.make_object_map(
                           params['data'], 
                           params['mask'], 
                           params['whitefield'], 
                           pixel_translations, 
                           u, 
                           subpixel=False)
    
    u0 = np.array(np.indices(params['mask'].shape))
    du = u-u0

    out = {'reference_image': O, 'n0': n0, 'm0': m0, 
           'pixel_map': u, 'pixel_map_residual': du, 
           'pixel_translations': pixel_translations, 
           'dfs': dfs, 'dss': dss}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)

    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/reference_image', file=f)


if __name__ == '__main__':
    main()
