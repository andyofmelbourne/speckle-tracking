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

import pyximport; pyximport.install()
import feature_matching as fm
import cmdline_config_cxi_reader

import numpy as np

if __name__ == '__main__':
    # get input 
    ###########
    args, params = cmdline_config_cxi_reader.get_all('stitch', 
                   'stitch frames together to form a merged view of the sample from projection images')
    params = params['stitch']
    
    if 'pixel_shifts' in params:
        uss, ufs = params['pixel_shifts']
    else :
        uss = ufs = np.zeros(params['whitefield'].shape, dtype=np.float64)
    
    # Do Stuff
    ##########

    # set masked pixels to negative 1
    mask             = params['mask'].astype(np.bool)
    params['frames'] = params['frames'].astype(np.float32)
    for i in range(len(params['frames'])):
        params['frames'][i][~mask] = -1
    
    params['whitefield'][~mask]    = -1 
    
    # add a regularization factor
    shape = params['whitefield'].shape
    reg   = params['reg']
    if params['reg'] is not None : 
        x = np.arange(shape[0]) - shape[0]//2
        x = np.exp( -x**2 / (2. * reg**2))
        y = np.arange(shape[1]) - shape[1]//2
        y = np.exp( -y**2 / (2. * reg**2))
        reg = np.outer(x, y)
    else :
        reg = 1
    
    # merge the frames
    atlas = fm.build_atlas_distortions(reg * params['frames'], 
                                       reg * params['whitefield'], 
                                       params['R_ss_fs'], 
                                       uss, ufs)
    
    # write the result 
    ##################
    out = {'O': atlas, 'reg': reg}
    cmdline_config_cxi_reader.write_all(params, args.filename, out)
    
    print('display: '+params['h5_group']+'/O') ; sys.stdout.flush()
    
