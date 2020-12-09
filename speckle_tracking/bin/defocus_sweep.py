#!/usr/bin/env python
import os
import tqdm
import speckle_tracking as st
from speckle_tracking import cmdline_config_cxi_reader
from speckle_tracking import cmdline_parser
from scipy.ndimage import gaussian_gradient_magnitude


import numpy as np
def defocus_sweep(z1_min, z1_max, N, z, data, mask, whitefield, basis, 
        x_pixel_size, y_pixel_size, translations, ls):
    """
    Sweep over possible defocus values
    """
    z1s = np.linspace(z1_min, z1_max, N)
    Os  = []

    it = tqdm.trange(z1s.shape[0], desc='sweeping defocus')
    
    for i in it:
        z1 = z1s[i]
        # generate pixel mapping
        pixel_map, pixel_translations, res = st.generate_pixel_map(
                    mask.shape, 
                    translations, 
                    basis, 
                    x_pixel_size, 
                    y_pixel_size, 
                    z, 
                    z1, 
                    None, 
                    None,
                    None,
                    verbose=False)
        
        # generate reference image
        I0 = st.make_object_map(data.astype(whitefield.dtype), mask, whitefield, pixel_translations, pixel_map, ls)[0]

        Os.append(np.mean(gaussian_gradient_magnitude(I0, sigma=ls)**2))

    z1 = z1s[np.argmax(Os)]
    return np.array(Os), z1


def main(overide={}):
    # get command line args and config
    sc  = 'defocus_sweep'
 
    # search the current directory for *.ini files if not present in cxi directory
    config_dirs = [os.path.split(os.path.abspath(__file__))[0]]
    
    # extract the first paragraph from the doc string
    des = defocus_sweep.__doc__.split('\n\n')[0]
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc, des, config_dirs=config_dirs, roi=True)
    params = params['defocus_sweep']
    
    # overide with input params (if any)
    params.update(overide)
    
    Os, z1 = defocus_sweep(params['z1_min'], 
                       params['z1_max'], 
                       params['n'], 
                       params['z'], 
                       params['data'], 
                       params['mask'], 
                       params['whitefield'], 
                       params['basis'], 
                       params['x_pixel_size'], 
                       params['y_pixel_size'], 
                       params['translations'],
                       params['ls'])
    
    out = {'Os': Os, 'defocus': z1}
    cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=True)

    # output display for gui
    with open('.log', 'w') as f:
        print('display: /'+params['h5_group']+'/Os', file=f)

if __name__ == '__main__':
    main()
