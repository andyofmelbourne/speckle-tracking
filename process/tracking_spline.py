import sys, os
root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.insert(0, os.path.join(root, 'utils'))

import config_reader
import cmdline_parser
import numpy as np
import h5py
import time

import pyximport; pyximport.install()
import feature_matching 
import poly_utils
import scipy.optimize

def minimise_R(atlas, frames, pos, frame_inds, pix_shifts, poly_coeffs=None):
    """
    given that:
        I_n(x) = O(x - x_n + ux(x))

    maximise:
        sum_n Pearson(I_n(x), O(x - x_n + u(x))) 
    
    with respect to x_n and u(x), where
        u(x, y) = sum_kk' a_kk' x^k y^k'
    
    and 
        ux(x, y) = du(x,y) / dx = sum_k=1 k' a_kk' x^(k-1) y^k'
        uy(x, y) = du(x,y) / dy = sum_kk'=1  a_kk' x^k y^(k'-1)

    and the x, y domain is set to -1 --> 1 over the roi

    keep: 
        a_00 no phase offset
        a_10 no x-shift
        a_01 no y-shift
        a_20 no x-scale
        a_02 no y-scale
    set to zero
    """
    from numpy.polynomial.polynomial import polyval2d
    
    # pixel indices
    shape = frames.shape[1:]
    i, j  = np.indices(shape)
    f_out = -np.ones( (len(frame_inds), shape[0], shape[1]), dtype=np.float)
    
    # offset the postions
    NN = len(frame_inds)
    pos = np.array(pos)
    pos[:, 0] += -np.max(pos[:, 0])+np.min(i + pix_shifts[0])
    pos[:, 1] += -np.max(pos[:, 1])+np.min(j + pix_shifts[1])
    
    pos2 = np.array(pos)[frame_inds]

    # centre pixel
    i0, j0 = -((1-shape[0])//2), -((1-shape[1])//2)
    
    # polyorder
    R  = np.ones((2, shape[0], shape[1]), dtype=np.float)
    K  = 5
    a  = np.zeros((K, K), dtype=np.float)
    ax = np.zeros((K, K), dtype=np.float)
    ay = np.zeros((K, K), dtype=np.float)
    
    mask = np.ones_like(a).astype(np.bool)
    mask[0, 0] = False # no phase offset
    mask[1, 0] = False # no x-shift
    mask[0, 1] = False # no y-shift
    mask[2, 0] = False # no x-scale
    mask[0, 2] = False # no y-scale
    
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    x, y = np.meshgrid(x, y, indexing='ij')

    def make_shifts_frames(xx, ax, ay, a, R):
        x_n, y_n = xx[:NN], xx[NN:2*NN]
        
        a[mask] = xx[2*NN:]
        
        ax = poly_utils.polyder2d(a, ax, 0)
        ay = poly_utils.polyder2d(a, ay, 1)
        
        # make pixel_shifts
        R[0] = polyval2d(x, y, ax)
        R[1] = polyval2d(x, y, ay)
        
        f_out.fill(-1)
        for k in range(NN):
            feature_matching.sub_pixel_warp(atlas, f_out[k], R, x_n[k], y_n[k])
        return R, f_out
    
    def fun(xx, ax, ay, a, R):
        R, f_out = make_shifts_frames(xx, ax, ay, a, R) 
        out = 0
        for fr, im in zip(f_out, frames):
            t = feature_matching.similarity_pearson(fr, im, i0, j0)
            out -= t
        print(out)
        return out
    
    import time
    pos2_mean = np.mean(pos2, axis=0)
    if poly_coeffs is None :
        x0 = np.concatenate((pos2[:,0], pos2[:, 1], np.zeros(K**2 - 5, dtype=np.float)))
    else :
        x0 = np.concatenate((pos2[:,0], pos2[:, 1], poly_coeffs[mask]))
    
    d0 = time.time()
    res = scipy.optimize.minimize(fun, x0, args=(ax, ay, a, R), tol = 1.0e-2, method='CG', \
                                  options={'disp' : True, 'maxiter' : 2, 'eps' : 1.0e-3})
    d1 = time.time()
    print(res)
    print('time to optimise:', d1-d0, 's') 
    print('fun0   :', fun(x0, ax, ay, a, R))   
    print('fun out:', fun(res.x, ax, ay, a, R)) ; sys.stdout.flush()
    
    R, f_out = make_shifts_frames(res.x, ax, ay, a, R) 

    a[mask] = res.x[2*NN:]


    pos2[:, 0] = pos2[:, 0] - np.mean(pos2[:, 0]) + pos2_mean[0]
    pos2[:, 1] = pos2[:, 1] - np.mean(pos2[:, 1]) + pos2_mean[1]
    posout = np.array(pos)
    posout[frame_inds] = pos2
    return R, f_out, a, posout

def minimise_pos(atlas, frames, pos, frame_inds, pix_shifts):
    """
    given that:
        I_n(x) = O(x - x_n + ux(x))

    maximise:
        sum_n Pearson(I_n(x), O(x - x_n + u(x))) 
    
    with respect to x_n and u(x), where
        u(x, y) = sum_kk' a_kk' x^k y^k'
    
    and 
        ux(x, y) = du(x,y) / dx = sum_k=1 k' a_kk' x^(k-1) y^k'
        uy(x, y) = du(x,y) / dy = sum_kk'=1  a_kk' x^k y^(k'-1)

    and the x, y domain is set to -1 --> 1 over the roi

    keep: 
        a_00 no phase offset
        a_10 no x-shift
        a_01 no y-shift
        a_20 no x-scale
        a_02 no y-scale
    set to zero
    """
    from numpy.polynomial.polynomial import polyval2d
    
    # pixel indices
    shape = frames.shape[1:]
    i, j  = np.indices(shape)
    f_out = -np.ones( (len(frame_inds), shape[0], shape[1]), dtype=np.float)
    
    # offset the postions
    NN = len(frame_inds)
    pos2 = np.array(pos)[frame_inds]
    pos2[:, 0] += -np.max(pos[:, 0])+np.min(i + pix_shifts[0])
    pos2[:, 1] += -np.max(pos[:, 1])+np.min(j + pix_shifts[1])
    pos2 = np.rint(pos2).astype(np.int)
    
    # centre pixel
    i0, j0 = -((1-shape[0])//2), -((1-shape[1])//2)
    
    x = np.linspace(-1, 1, shape[0])
    y = np.linspace(-1, 1, shape[1])
    x, y = np.meshgrid(x, y, indexing='ij')

    def make_shifts_frames(xx):
        x_n, y_n = xx[:NN], xx[NN:]
        
        f_out.fill(-1)
        for k in range(NN):
            feature_matching.sub_pixel_warp(atlas, f_out[k], pix_shifts, x_n[k], y_n[k])
        return f_out
    
    def fun(xx):
        f_out = make_shifts_frames(xx) 
        out = 0
        for fr, im in zip(f_out, frames):
            t = feature_matching.similarity_pearson(fr, im, i0, j0)
            out -= t
        print(out)
        return out
    
    import time
    x0 = np.concatenate((pos2[:,0], pos2[:, 1]))
                
    d0 = time.time()
    res = scipy.optimize.minimize(fun, x0, tol = 1.0e-2, method='CG', \
                                  options={'disp' : True, 'maxiter' : 1, 'eps' : 1.0e-5})
    d1 = time.time()
    print(res)
    print('time to optimise:', d1-d0, 's') 
    print('fun0   :', fun(x0))   
    print('fun out:', fun(res.x)) ; sys.stdout.flush()
    
    f_out = make_shifts_frames(res.x) 

    return f_out

def forward_frames(atlas, R, pos, frame_inds, shape):
    shape = R.shape[1:]
    i, j  = np.indices(shape)
    f_out = -np.ones( (len(frame_inds), shape[0], shape[1]), dtype=np.float)
    
    # offset the postions
    pos2 = np.array(pos)
    pos2[:, 0] += -np.max(pos[:, 0])+np.min(i + R[0])
    pos2[:, 1] += -np.max(pos[:, 1])+np.min(j + R[1])
    pos2 = pos2.astype(np.int)
    
    for kk, k in enumerate(frame_inds) :
        print( pos2[k])
        ss = i - pos2[k][0] + R[0]
        fs = j - pos2[k][1] + R[1]
        ss = np.rint(ss).astype(np.int)
        fs = np.rint(fs).astype(np.int)
        mask = (ss > 0) * (ss < atlas.shape[0]) * (fs > 0) * (fs < atlas.shape[1])
        
        f_out[kk][mask] =  atlas[ss[mask], fs[mask]]

    return f_out

def grid_refine(atlas, frames, pos, frame_inds, pix_shifts, grid, w):
    # make a grid 
    pass


if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('tracking_spline', 'parameterised refinement of pixel shifts')

    f = h5py.File(args.filename, 'r')
    in_h5, fnam = config_reader.config_read_from_h5(args.config, \
                                                    f, extract=True, roi=True)
    in_h5       = in_h5['tracking_spline']
    shape       = f[params['data']].shape
    atlas       = f[params['atlas']][()]

    f.close()
    
    roi = in_h5['roi']
    w   = in_h5['window']
    roi = in_h5['roi']
    g   = in_h5['grid']
    
    W      = in_h5['whitefield']
    mask   = in_h5['mask']
    W[W==0]  = 1
    frames = in_h5['data'][in_h5['good_frames']]
    Os     = mask * np.array(frames).astype(np.float) / W.astype(np.float)
    frames = mask * np.array(frames).astype(np.float) 
    
    if in_h5['pix_positions'].shape[1] > 2 :
        pos  = np.zeros((shape[0], 2), dtype=np.float)
        pos[in_h5['pix_positions'][:, 0].astype(np.int)] = in_h5['pix_positions'][:, 1:]
    else :
        pos             = in_h5['pix_positions']
    
    # set the masked values to -1
    frames[frames==0] = -1
    Os[Os==0]         = -1
    atlas[atlas==0]   = -1
    W[~mask]          = -1

    print(frames.shape, Os.shape, Os[in_h5['ref_frames']].shape)
    R, f_out, pcoeffs, pos = minimise_R(atlas, Os[in_h5['ref_frames']], 
                                        pos, in_h5['ref_frames'], 
                                        in_h5['pixel_shifts'], in_h5['poly_coeffs'])

    atlas_out = feature_matching.build_atlas_distortions(frames, W, 
                             pos[in_h5['good_frames']], R[0], R[1])
    
    config_reader.write_h5(args.filename, params['h5_group'], 
                          {'frame_comp' : np.array([Os[5], f_out[5]])})
    print('display: '+params['h5_group']+'/frame_comp') ; sys.stdout.flush()

    ashape = (2, min(atlas.shape[0], atlas_out.shape[0]), 
                 min(atlas.shape[1], atlas_out.shape[1]))
    atlas_comp = np.array([atlas[:ashape[1], :ashape[2]], 
                           atlas_out[:ashape[1], :ashape[2]]])
    Rout = np.zeros((2,)+shape[1:], dtype=np.float)
    Rout[:, roi[0]:roi[1], roi[2]:roi[3]] = R

    config_reader.write_h5(args.filename, params['h5_group'], 
                          {'frames_forward' : f_out, 
                           'R' : Rout, 
                           'atlas' : atlas_out, 
                           'atlas_comp' : atlas_comp,
                           'poly_coeffs' : pcoeffs, 
                           'pix_positions' : pos})
    
    # anoying but need to do this to give the 
    # widget time to display
    time.sleep(1)
