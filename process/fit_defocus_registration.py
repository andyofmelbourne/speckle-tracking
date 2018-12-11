"""
for T=e^{i phi}
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

import pyximport; pyximport.install()
import feature_matching as fm
import cmdline_config_cxi_reader
import cmdline_parser 

import numpy as np
rank = 0

def make_shifts(dz, shape, params):
    z  =  params['distance']
    z1 = params['defocus']
    z2 = params['distance']-params['defocus']
    M  = (z1+z2)/z1
    ze = z2/M
    
    i, j = np.ogrid[:shape[0]:, :shape[1]:] 
    i = i - np.mean(i)
    j = j - np.mean(j)
    uss = i * ze * (1/(z+dz[0]) - 1/z) / params['y_pixel_size'] + 0*j
    ufs = j * ze * (1/(z+dz[1]) - 1/z) / params['x_pixel_size'] + 0*i
    return uss, ufs



def fit(frames, W, steps, roi, params, callback=None):
    if callback is None :
        callback = lambda x : None
    
    # the regular pixel values
    i, j  = np.ogrid[0:W.shape[-2], 0:W.shape[-1]]
    
    # offset the steps
    steps2 = np.array(steps)
    steps2 += fm.steps_offset(steps, np.zeros((2,) + W.shape, dtype=np.int))
    
    # now offset the steps so that 0,0 is in the centre of the 
    # roi
    N = (roi[1]-roi[0])
    M = (roi[3]-roi[2])
    steps2[:, 0] += roi[0]
    steps2[:, 1] += roi[2]

    atlas   = np.zeros((N, M), dtype=np.float)
    overlap = np.zeros((N, M), dtype=np.float)
    
    #uss, ufs = np.rint(pixel_shifts).astype(np.int)
    WW       = W**2 
    
    # sharpen along fs
    out = []
    variance = []
    dzs = np.linspace(params['defocus_range'][0], params['defocus_range'][1], 100)
    for dz in dzs:
        uss, ufs = make_shifts((0, dz), W.shape, params)
        atlas.fill(0)
        overlap.fill(0)
        atlas = build_atlas_sub(frames, W, WW, steps2, uss, ufs, i, j, atlas, overlap)
        variance.append(np.var(atlas[atlas>0]))
        
        out.append(atlas.copy())
        #callback(atlas)
    
    defocus_fs = dzs[np.argmax(variance)]
    
    # sharpen along ss
    variance = []
    for dz in dzs:
        uss, ufs = make_shifts((dz, defocus_fs), W.shape, params)
        atlas.fill(0)
        overlap.fill(0)
        atlas = build_atlas_sub(frames, W, WW, steps2, uss, ufs, i, j, atlas, overlap)
        variance.append(np.var(atlas[atlas>0]))
        
        out.append(atlas.copy())
        #callback(atlas)
    
    defocus_ss = dzs[np.argmax(variance)]
    print('defocus along ss / fs (best fit):', defocus_ss, defocus_fs)
    
    uss, ufs = make_shifts((defocus_ss, defocus_fs), W.shape, params)
    atlas.fill(0)
    overlap.fill(0)
    atlas = build_atlas_sub(frames, W, WW, steps2, uss, ufs, i, j, atlas, overlap)
    out.append(atlas.copy())
    
    return np.array(out), uss, ufs

def build_atlas_sub(frames, W, WW, steps, uss, ufs, i, j, atlas, overlap):
    """
    assume:
        frames_i(x) = atlas(x + u(x) - x_i)  W(x)
    
    atlas: 
        atlas(x) = sum_i W(x'_i) frames_i(x'_i) / sum_i W^2(x'_i)
    
    where:
        x - x'_i - u(x'_i) + x_i = 0
    
    bad pixels are < 0
    """
    N, M = atlas.shape
    # build the atlas and overlap map frame by frame
    for frame, step in zip(frames, steps):
        mask0 = frame>0
        ss = np.rint(i + uss - step[0]).astype(np.int)
        fs = np.rint(j + ufs - step[1]).astype(np.int)
        mask = (ss > 0) * (ss < N) * (fs > 0) * (fs < M) * mask0
        atlas[  ss[mask], fs[mask]] += (frame*W)[mask]
        overlap[ss[mask], fs[mask]] += WW[mask] 
    
    bad          = (overlap < 2.0)
    atlas[bad]   = -1 
    atlas[~bad] /= (overlap[~bad] + 1.0e-5)
    return atlas
    

if __name__ == '__main__':
    # get input 
    ###########
    # get command line args and config
    sc  = 'fit_defocus_registration'
    des = 'fit the sample to focus distance (and astigmatism) by registration of speckles.'
    args, params = cmdline_parser.parse_cmdline_args(sc, des)
    
    # now load the necessary data
    args, params = cmdline_config_cxi_reader.get_all(sc,des)
    params = params['fit_defocus_registration']

    if ('pixel_shifts' in params) and (params['pixel_shifts'] is not None) :
        pix_shifts = params['pixel_shifts']
    else :
        pix_shifts = np.zeros((2,) + params['whitefield'].shape[-2:], dtype=np.float64)

    z1 = params['defocus']
    z2 = params['distance']-params['defocus']
    M  = (z1+z2)/z1
    ze = z2/M
    print('mag        :', M)
    print('z effective:', ze)
    
    # Do Stuff
    ##########
    # set masked pixels to negative 1
    mask              = params['mask'].astype(np.bool)
    params['frames']  = params['frames'][params['good_frames']].astype(np.float64)
    for i in range(len(params['frames'])):
        params['frames'][i][~mask] = -1
    
    if len(params['whitefield'].shape) == 3 :
        for i in range(len(params['whitefield'])):
            params['whitefield'][i][~mask]    = -1 
    else :
        params['whitefield'][~mask]    = -1 
    
    atlas, uss, ufs = fit(params['frames'], params['whitefield'], 
                params['R_ss_fs'], params['o_roi'], 
                params)
    
    # write the result 
    ##################
    out = {'O': atlas, 'pixel_shifts': np.array([uss, ufs])}
    cmdline_config_cxi_reader.write_all(params, args.filename, out)
    
    import time
    for i in range(3):
        print('display: '+params['h5_group']+'/O') ; sys.stdout.flush()
        time.sleep(1)


