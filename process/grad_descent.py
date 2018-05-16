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

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.Get_rank()
size = comm.Get_size()


def mk_reg(shape, reg):
    if reg is not None : 
        x = np.arange(shape[0]) - shape[0]//2
        x = np.exp( -x**2 / (2. * reg**2))
        y = np.arange(shape[1]) - shape[1]//2
        y = np.exp( -y**2 / (2. * reg**2))
        reg = np.outer(x, y)
    else :
        reg = 1
    return reg

def forward_frames(atlas, I, W, R, u):
    # the regular pixel values
    ij  = np.ogrid[0:W.shape[0], 0:W.shape[1]]
    frames = []
    for frame, step in zip(I.arr, R.arr):
        If = W*fm.forward_frame(atlas, W, step, u, ij, sub_pixel = True)
        
        frames.append(If.copy())
    return np.array(frames)

def euclid_err_MpiArray(atlas, I, W, R, u, return_pix_map = False):
    # the regular pixel values
    ij  = np.ogrid[0:W.shape[0], 0:W.shape[1]]
    
    # \sum_i Sim(I_i, I_for, u)
    err   = np.zeros_like(I.arr[0])
    terms = np.zeros_like(I.arr[0], dtype=np.int)
    for frame, step in zip(I.arr, R.arr):
        If = fm.forward_frame(atlas, W, step, u, ij, sub_pixel = True)
        m  = (If>0)*(frame>0)*(W>0)
        err[m] += (If[m] - frame[m]/W[m])**2 
        terms  += m
    
    if return_pix_map :
        err   = comm.reduce(err)
        terms = comm.reduce(terms)
        if rank == 0 :
            terms[terms==0] = 1
            return err / terms 
        else :
            return None
    else :
        err   = comm.allreduce([np.sum(err)])[0]
        terms = comm.allreduce([np.sum(terms)])[0]
        return err / terms 
    

def line_plot(atlas, I, W, R, u, du, N = 21, bounds = [-20, 20], max_iters = 100):
    """
    assume that the pix positions are already offset
    """
    # scale du 
    du_max = comm.allreduce([np.abs(du).max()], op=MPI.MAX)[0]
    du = du / du_max
    
    def fun(step):
        err = euclid_err_MpiArray(atlas, I, W, R, u + step*du)
        if rank==0 : print(step, err)
        return err
    
    steps = np.linspace(bounds[0], bounds[1], N)
    errs  = np.array([fun(s) for s in steps])
    
    return steps, errs

def line_search(atlas, I, W, R, u, du, bounds = [-20, 0], max_iters = 100):
    """
    assume that the pix positions are already offset
    """
    # scale du 
    du_max = np.abs(du).max()
    du = du / du_max
    
    def fun(step):
        err = euclid_err_MpiArray(atlas, I, W, R, u + step*du)
        if rank == 0 : print(step, err)
        return err
    
    from scipy.optimize import minimize_scalar
    res = minimize_scalar(fun, bounds = bounds, method='bounded',
                          options = {'maxiter' : max_iters, 'xatol' : 0.1})
    if rank == 0 : print(res)
    u += res.x * du
    return u

def mk_grad_u_MpiArray(atlas, I, W, R, u):
    """
    Assume that frames, W, pixel_shifts are MpiArray objects split along 
    the ss axis of the detector.
    
    assume that the pix positions are already offset
    """
    # the regular pixel values
    ij  = np.ogrid[0:W.shape[0], 0:W.shape[1]]
    
    grad_u  = np.zeros_like(u, dtype=np.float64)
    If      = np.zeros(u.shape[1:], dtype=np.float64)
    grad_ov = np.zeros(u.shape[1:], dtype=np.float64)
    
    at_mask = atlas>=0
    
    # avoid pixels to the left/right top/bottom of at_mask
    at_mask *= np.roll(at_mask, 1,  0)
    at_mask *= np.roll(at_mask, -1, 0)
    at_mask *= np.roll(at_mask, 1,  1)
    at_mask *= np.roll(at_mask, -1, 1)
    
    atlas_grad = np.gradient(atlas, axis=(0,1))
    
    for frame, step in zip(I.arr, R.arr):
        # make the forward model for the frame
        If[:] = fm.forward_frame(atlas, W, step, u, ij, sub_pixel = True)
        dss   = fm.forward_frame(atlas_grad[0], W, step, u, ij, sub_pixel = True)
        dfs   = fm.forward_frame(atlas_grad[1], W, step, u, ij, sub_pixel = True)
        mask  = fm.forward_frame(at_mask.astype(np.float), W, step, u, ij, sub_pixel = True)
        mask  = (mask > 0.8) * (If>0) * (frame>0) 
        
        # these derivatives are now masked arrays
        grad_u[0] += dss * (W * If - frame) * mask 
        grad_u[1] += dfs * (W * If - frame) * mask
        grad_ov   += mask
    
    # allreduce and normalise the grads
    if rank==0 :
        grad_u0  = np.empty(grad_u.shape, grad_u.dtype)
        grad_ov0 = np.empty(grad_ov.shape, grad_ov.dtype)
    else :
        grad_u0  = None
        grad_ov0 = None
    
    dtype = I.numpy_to_mpi_dtype(grad_u.dtype)
    comm.Reduce([grad_u, dtype],  [grad_u0, dtype], root=0)
    
    dtype = I.numpy_to_mpi_dtype(grad_ov.dtype)
    comm.Reduce([grad_ov, dtype], [grad_ov0, dtype], root=0)

    if rank == 0 :
        # normalise the gradient
        bad           = grad_ov0 < 0.8
        grad_u[0][~bad] = grad_u0[0][~bad] / grad_ov0[~bad]
        grad_u[1][~bad] = grad_u0[1][~bad] / grad_ov0[~bad]
        grad_u[0][bad]  = 0.
        grad_u[1][bad]  = 0.
        
    comm.Bcast([grad_u, dtype], root=0)
    
    return grad_u

def build_atlas_distortions_MpiArray(frames, W, steps, pixel_shifts, reg = 1, return_steps = False, weights=None):
    """
    Assume that frames, W, pixel_shifts are MpiArray objects split along 
    the ss axis of the detector.

    The steps are offset such that:
        min(i + pixel_shifts[0] - int(round(step[0]))) = 0
    
    And the atlas shape is given by: 
        N = max(i + pixel_shifts[0] - int(round(step[0])))
    """
    # determine the atlas shape
    i, j  = np.ogrid[0:W.shape[0], 0:W.shape[1]]
    
    # offset steps 
    off0   = np.min(i + pixel_shifts[0]) - comm.allreduce([np.max(steps.arr[:, 0])], op=MPI.MAX)[0]
    off1   = np.min(j + pixel_shifts[1]) - comm.allreduce([np.max(steps.arr[:, 1])], op=MPI.MAX)[0]
    steps.arr = steps.arr + np.array([off0, off1])
    
    # define the atlas grid
    N = np.max(i + pixel_shifts[0]) - comm.allreduce([np.min(steps.arr[:, 0])], op=MPI.MIN)[0]
    M = np.max(j + pixel_shifts[1]) - comm.allreduce([np.min(steps.arr[:, 1])], op=MPI.MIN)[0]
    N = math.ceil(N)
    M = math.ceil(M)
    
    atlas, overlap = fm.build_atlas(reg*frames.arr, reg*W, 
                                    steps.arr, pixel_shifts, 
                                    offset_steps = False, 
                                    return_steps = False, 
                                    return_overlap = True,
                                    atlas_shape = (N, M),
                                    sub_pixel = True, 
                                    weights = weights)
    
    # allreduce the atlas
    if rank==0 :
        atlas0   = np.empty(atlas.shape  , atlas.dtype)
        overlap0 = np.empty(overlap.shape, overlap.dtype)
    else :
        atlas0   = None
        overlap0 = None
    
    dtype = frames.numpy_to_mpi_dtype(atlas.dtype)
    comm.Reduce([atlas, dtype], [atlas0, dtype], root=0)
    
    dtype = frames.numpy_to_mpi_dtype(overlap.dtype)
    comm.Reduce([overlap, dtype], [overlap0, dtype], root=0)
    
    if rank == 0 :
        bad   = overlap0 < (1.0e-2 * W.max())
        atlas[~bad] = atlas0[~bad] / overlap0[~bad]
        atlas[bad]  = -1
    
    comm.Bcast([atlas, dtype], root=0)
    
    if return_steps :
        return atlas, steps
    else :
        return atlas

def get_input():
    args, params = cmdline_config_cxi_reader.get_all('grad_descent', 
                   'update the pixel shifts according to a least squares minimisation procedure',
                   exclude=['frames'])
    params = params['grad_descent']
    
    # frames, split by frame no.
    roi = params['roi']
    roi = [params['good_frames'], slice(roi[0],roi[1]), slice(roi[2],roi[3])]
    params['frames']     = MpiArray_from_h5(args.filename, params['frames'], 
                                            axis=0, dtype=np.float64, roi=roi)
    #params['frames'] = params['frames'][params['good_frames']]
    
    if rank != 0 :
        params['R_ss_fs']    = None 
    
    params['R_ss_fs'] = MpiArray(params['R_ss_fs'])
    params['R_ss_fs'].scatter(axis=0)
     
    # set masked pixels to negative 1
    for i in range(params['frames'].arr.shape[0]):
        params['frames'].arr[i][~params['mask']] = -1
    
    params['whitefield'][~params['mask']]    = -1 
    
    if params['pixel_shifts'] is None :
        params['pixel_shifts'] = np.zeros((2,) + params['whitefield'].shape, dtype=np.float64)
    
    # add a regularization factor
    shape = params['whitefield'].shape
    reg   = mk_reg(shape, params['reg'])
    
    return args, params, reg

if __name__ == '__main__':
    args, params, reg = get_input()

    print(params['R_ss_fs'].shape)
    print(params['frames'].shape)
    
    # merge the frames
    if params['atlas'] is None :
        atlas, params['R_ss_fs'] = build_atlas_distortions_MpiArray(params['frames'], 
                                                  params['whitefield'], 
                                                  params['R_ss_fs'], 
                                                  params['pixel_shifts'], 
                                                  reg=reg,
                                                  return_steps = True)
    else :
        atlas = params['atlas']
    
    if params['atlas_smooth'] is not None and params['atlas_smooth'] is not 0 :
        atlas = gaussian_filter(atlas, params['atlas_smooth'])
    
    for i in range(params['max_iters']):
        grad_u = mk_grad_u_MpiArray(atlas, params['frames'], params['whitefield'], 
                                    params['R_ss_fs'], params['pixel_shifts'],
                                    )
        
        params['pixel_shifts']  = line_search(atlas, params['frames'], params['whitefield'], 
                                              params['R_ss_fs'], params['pixel_shifts'], 
                                              grad_u, bounds=[-params['max_step'], 0])
        
        # smooth pix shifts
        if params['pix_reg'] is not None :
            params['pixel_shifts'][0] = gaussian_filter(params['pixel_shifts'][0], params['pix_reg'])
            params['pixel_shifts'][1] = gaussian_filter(params['pixel_shifts'][1], params['pix_reg'])
        
        # real-time output
        if rank == 0 :
            out = {'pixel_shifts' : params['pixel_shifts']}
            cmdline_config_cxi_reader.write_all(params, args.filename, out)
            print('display: '+params['h5_group']+'/pixel_shifts') ; sys.stdout.flush()

    err_pix_map = euclid_err_MpiArray(atlas, params['frames'], params['whitefield'], 
                                        params['R_ss_fs'], params['pixel_shifts'], 
                                        return_pix_map=True)
    
    atlas  = build_atlas_distortions_MpiArray(params['frames'], 
                                              params['whitefield'], 
                                              params['R_ss_fs'], 
                                              params['pixel_shifts'], 
                                              reg=reg)
    
    if rank == 0 :
        out = {'err_pix_map' : err_pix_map, 
               'pixel_shifts' : params['pixel_shifts'], 
               'atlas' : atlas, 'grad_pix': grad_u}
        cmdline_config_cxi_reader.write_all(params, args.filename, out)

