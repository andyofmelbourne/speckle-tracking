#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

def main(overide={}):
    # get command line args and config
    sc  = 'fit_thon_rings'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.fit_thon_rings.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params['fit_thon_rings']

    # overide with input params (if any)
    params.update(overide)
    
    z1, res = st.fit_thon_rings(
                             params['data'],
                             params['x_pixel_size'],
                             params['y_pixel_size'],
                             params['distance'],
                             params['wavelength'],
                             params['mask'],
                             params['whitefield'],
                             None,
                             None,
                             params['sigma'],
                             params['edge_pix'],
                             params['window'],
                             params['rad_range']
                             )

    out = {'defocus': z1}
    out.update(res)
    
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/thon_display', file=f)

if __name__ == '__main__':
    main()
