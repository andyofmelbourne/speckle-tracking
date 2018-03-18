import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))

import config_reader
import cmdline_parser
import numpy as np
import h5py

def speckle_error(frames, atlas, pixel_shifts, window, ij_grid):
    pass





if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('tracking_spline', 'parameterised refinement of pixel shifts')
    
    f = h5py.File(args.filename, 'r')
    h5_params, fnam = config_reader.config_read_from_h5(args.config, f)
    f.close()

    # make the ij grid within the roi:
    w   = params['window']
    roi = params['roi']
    g   = params['grid']
    i = np.ogrid[w[0]//2 : roi[1]-roi[0]-w[0]//2 : g[0]*1J]
    j = np.ogrid[w[1]//2 : roi[3]-roi[2]-w[1]//2 : g[1]*1J]
    
    i, j = np.meshgrid(i, j, indexing='ij')
    ij_grid = np.vstack((i.ravel(), j.ravel()))

    print('display:', params['atlas']) ; sys.stdout.flush()
    
    # anoying but need to do this to give the 
    # widget time to display
    import time
    time.sleep(1)
