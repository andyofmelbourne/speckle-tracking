"""
"""

#! /usr/bin/env python
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals

# make an example cxi file
# with a small sample and small aberations

import sys, os
base = os.path.join(os.path.dirname(__file__), '..')
root = os.path.abspath(base)
sys.path.insert(0, os.path.join(root, 'utils'))

#import pyximport; pyximport.install()
import cmdline_config_cxi_reader

import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def get_input():
    args, params = cmdline_config_cxi_reader.get_all('make_whitefield', 
                   'estimate the whitefield by taking the median value of every pixel')
    params = params['make_whitefield']
    
    return args, params

if __name__ == '__main__':
    args, params = get_input()

    W = np.median(params['frames'], axis=0)

    st = params['sigma_t']
    Ws = np.empty(params['frames'].shape, dtype=np.float)
    if st is not None and st is not 0 : 
        for i in range(Ws.shape[0]):
            i_min = max(i-st, 0)
            i_max = min(i+st, Ws.shape[0])
            Ws[i] = np.median(params['frames'][i_min:i_max], axis=0)
                
            if i % 10 == 0 :
                out = {'whitefield' : Ws[i]}
                cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)
                print('display: '+params['h5_group']+'/whitefield') ; sys.stdout.flush()
    else :
        Ws[:] = W
    
    # real-time output
    if rank == 0 :
        out = {'whitefield' : W, 'whitefields' : Ws}
        cmdline_config_cxi_reader.write_all(params, args.filename, out, apply_roi=False)
        print('display: '+params['h5_group']+'/whitefield') ; sys.stdout.flush()
