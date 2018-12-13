"""
I need to make get_all mpi safe
split the data according to pixel index
need a new build_atlas_distortions for sub pixel regions
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
import feature_matching as fm
import cmdline_config_cxi_reader
from mpiarray import MpiArray
from mpiarray import MpiArray_from_h5

import math
import numpy as np
import h5py
from scipy.ndimage.filters import gaussian_filter
import itertools

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()

def get_input():
    """
    Blah 

    Parameters:
    ===========
    """
    args, params = cmdline_config_cxi_reader.get_all('update_pixel_map', 
                   'update the pixel shifts according to a least squares minimisation procedure',
                   exclude=['frames', 'whitefield', 'mask'])
    params = params['update_pixel_map']
    
    # split by ss pixels 
    ####################
    
    # special treatment for frames
    roi = params['roi']
    roi = (params['good_frames'], slice(roi[0],roi[1]), slice(roi[2],roi[3]))
    
    # easy for the mask
    params['frames']     = MpiArray_from_h5(args.filename, params['frames'], axis=1, roi=roi, dtype=np.float64)
    params['mask']       = MpiArray_from_h5(args.filename, params['mask']  , axis=0, roi=roi[1:], dtype=np.bool)
    
    with h5py.File(args.filename, 'r') as f:
        shape = f[params['whitefield']].shape
    if len(shape) == 2 :
        params['whitefield']  = MpiArray_from_h5(args.filename, params['whitefield'], axis=0, roi=roi[1:], dtype=np.float64)
        params['whitefield'].arr[~params['mask'].arr]    = -1 
    else :
        params['whitefield']  = MpiArray_from_h5(args.filename, params['whitefield'], axis=1, roi=roi, dtype=np.float64)
        for i in range(params['whitefield'].arr.shape[0]):
            params['whitefield'].arr[i][~params['mask'].arr] = -1
    
    # set masked pixels to negative 1
    for i in range(params['frames'].arr.shape[0]):
        params['frames'].arr[i][~params['mask'].arr] = -1

    # offset positions
    if params['pixel_shifts'] is not None :
        params['R_ss_fs'] += fm.steps_offset(params['R_ss_fs'], params['pixel_shifts'])

    # special treatment for the pixel_shifts
    if params['pixel_shifts'] is None and rank == 0 :
        params['pixel_shifts'] = np.zeros((2,) + params['whitefield'].shape[-2:], dtype=np.float64)
    
    if rank != 0 :
        params['pixel_shifts'] = None
    
    params['pixel_shifts'] = MpiArray(params['pixel_shifts'])
    params['pixel_shifts'].scatter(axis=1)
    
    return args, params

def grid_search_ss_split(atlas, I, W, R, u, bounds=[-20, 20]):
    ss_offset = W.distribution.offsets[rank] 
    print('rank, W.shape, offset:', rank, W.shape, W.arr.shape, ss_offset, W.distribution.offsets[rank])
    
    def fun(ui, uj, i, j, w): 
        ss   = np.rint(-R[:,0] + ui + i + ss_offset).astype(np.int)
        fs   = np.rint(-R[:,1] + uj + j).astype(np.int)
        m    = (ss>0)*(ss<atlas.shape[0])*(fs>0)*(fs<atlas.shape[1])
        forw = atlas[ss[m], fs[m]]
        m1   = (forw>0) * (w[m, i, j] > 0)
        n   = np.sum(m1)
        if n > 0 :
            #err = np.sum(m1*(forw - I.arr[m, i, j]/w[m, i, j])**2 ) / n
            err = np.sum(m1*(forw*w[m, i, j] - I.arr[m, i, j])**2 ) / n
        else :
            err = 1e100
        return err
     
    # define the search window
    k = np.arange(int(bounds[0]), int(bounds[1])+1, 1)
    k, l = np.meshgrid(k, k, indexing='ij')
    kls  = np.vstack((k.ravel(), l.ravel())).T
    
    # define the pupil idices
    ss = np.arange(W.arr.shape[-2])
    fs = np.arange(W.arr.shape[-1])

    if len(W.arr.shape) == 2 :
        w = np.array([W.arr for i in range(I.shape[0])])
    else :
        w = W.arr
    
    ss_min = comm.allreduce(np.max(ss), op=MPI.MIN)
    errs  = np.empty((len(kls),), dtype=np.float64)
    u_out = MpiArray(u.arr, axis=1)
    for i in ss :
        print(rank, i, i+ss_offset) ; sys.stdout.flush()
        for j in fs :
            errs.fill(1e100)
            for k, kl in enumerate(kls) :
                if np.any(w[:, i, j]) > 0 :
                    errs[k] = fun(u.arr[0, i, j] + kl[0], u.arr[1, i, j] + kl[1], i, j, w)
                
            k = np.argmin(errs) 
            u_out.arr[:, i, j] += kls[k]
        
        # output every 10 ss pixels
        if i % 1 == 0 and i <= ss_min and callback is not None :
            ug = u_out.gather()
            callback(ug)

    return u_out

def grid_search_ss_split_sub_pix(atlas, I, W, R, u, bounds=[-1, 1], max_iters = 100):
    ss_offset = W.distribution.offsets[rank] 
    
    def fun(x, i = 0, j = 0, w = 0, uu = 0):
        forw = np.empty((R.shape[0],), dtype=np.float)
        fm.sub_pixel_atlas_eval(atlas, forw, -R[:,0] + x[0] + uu[0] + i + ss_offset
                                           , -R[:,1] + x[1] + uu[1] + j)
        
        m   = (forw>0) * (w > 0)
        n   = np.sum(m)
        if n > 0 :
            err = np.sum((forw[m]*w[m] - I.arr[m, i, j])**2 ) / n
        else :
            err = -1
        return err

    if len(W.arr.shape) == 2 :
        w = np.array([W.arr for i in range(I.shape[0])])
    else :
        w = W.arr
    
    # define the pupil idices
    ss = np.arange(W.arr.shape[-2])
    fs = np.arange(W.arr.shape[-1])
    
    ss_min = comm.allreduce(np.max(ss), op=MPI.MIN)
    u_out = MpiArray(u.arr, axis=1)
    for i in ss :
        for j in fs :
            x0   = np.array([0., 0.])
            err0 = fun(x0, i, j, w[:, i, j], u.arr[:, i, j])
            options = {'maxiter' : max_iters, 'eps' : 0.1, 'xatol': 0.1}
            from scipy.optimize import minimize
            res = minimize(fun, x0, bounds = [bounds, bounds], 
                           args = (i, j, w[:, i, j], u.arr[:, i, j]),
                           options = {'maxiter' : max_iters, 'eps' : 0.1, 'xatol': 0.1})
            if rank == 0 and j==fs[fs.shape[0]//2]: print(err0, res); sys.stdout.flush()
            if res.fun > 0 :
                u_out.arr[:, i, j] += res.x
        
        # output every 10 ss pixels
        if i % 1 == 0 and i <= ss_min and callback is not None :
            ug = u_out.gather()
            callback(ug)
    
    return u_out

if __name__ == '__main__':
    args, params = get_input()
    
    # provide real time feedback
    def callback(u):
        if rank == 0 :
            out = {'pixel_shifts' : u}
            cmdline_config_cxi_reader.write_all(params, args.filename, out)
            print('display: '+params['h5_group']+'/pixel_shifts') ; sys.stdout.flush()
    
    if params['max_step']>1 :
        params['pixel_shifts']  = grid_search_ss_split(params['atlas'], params['frames'], params['whitefield'], 
                                              params['R_ss_fs'], params['pixel_shifts'], 
                                              bounds=[-params['max_step'], params['max_step']])
    if params['sub_pixel'] is True :
        params['pixel_shifts']  = grid_search_ss_split_sub_pix(params['atlas'], params['frames'], 
                                              params['whitefield'], params['R_ss_fs'], 
                                              params['pixel_shifts'], bounds=[-1.0, 1.0])
                                              #bounds=[-params['max_step'], params['max_step']])
    u = params['pixel_shifts'].gather()
    callback(u)
