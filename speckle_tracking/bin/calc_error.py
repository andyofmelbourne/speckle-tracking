#!/usr/bin/env python
import os
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser 

def main(overide={}):
    print('Running...')
    # get command line args and config
    sc  = 'calc_error'
     
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = st.calc_error.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params['calc_error']

    # overide with input params (if any)
    params.update(overide)
    
    error_total, error_frame, error_pixel, error_residual, error_reference, norm, flux_correction, res = st.calc_error(
            params['data'], 
            params['mask'], 
            params['whitefield'], 
            params['pixel_translations'], 
            params['reference_image'], 
            params['pixel_map'], 
            params['n0'], 
            params['m0'], 
            subpixel=True, verbose=False)
    
    out = {'error_total': error_total, 
           'error_frame': error_frame, 
           'error_pixel': error_pixel, 
           'error_residual': error_residual, 
           'error_reference': error_reference, 
           'error_norm': norm,
           'flux_correction': flux_correction}
    
    if '1d_data_vs_forward' in res :
        out['1d_data_vs_forward'] = res['1d_data_vs_forward']
    else :
        out['forward'] = res['forward']
    
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)
    
    # output display for gui
    with open('.log', 'w') as f:
        print('display: '+params['h5_group']+'/error_pixel', file=f)


if __name__ == '__main__':
    main()
