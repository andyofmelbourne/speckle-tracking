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

def speckle_error(frames, atlas, pixel_shifts, window, ij_grid):
    pass

def minimise_R_old(atlas, frames, pos, frame_inds, pix_shifts):
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
            ss = i - x_n[k] + R[0]
            fs = j - y_n[k] + R[1]
            ss = np.rint(ss).astype(np.int)
            fs = np.rint(fs).astype(np.int)
            m = (ss > 0) * (ss < atlas.shape[0]) * (fs > 0) * (fs < atlas.shape[1])
            
            f_out[k][m] =  atlas[ss[m], fs[m]]
        return R, f_out
    
    def fun(xx, ax, ay, a, R):
        R, f_out = make_shifts_frames(xx, ax, ay, a, R) 
        out = 0
        for fr, im in zip(f_out, frames):
            t = feature_matching.similarity_pearson(fr, im, i0, j0)
            out -= t
            print(t)
        return out
    
    import time
    x0 = np.concatenate((pos2[:,0], pos2[:, 1], np.zeros(K**2 - 5, dtype=np.float)))

    d0 = time.time()
    res = scipy.optimize.minimize(fun, x0, args=(ax, ay, a, R), options={'disp' : True})
    d1 = time.time()
    print(res)
    print('time to optimise:', d1-d0, 's') 
    print('fun0   :', fun(x0, ax, ay, a, R))
    print('fun out:', fun(res.x, ax, ay, a, R))

    R, f_out = make_shifts_frames(res.x, ax, ay, a, R) 
    return R, f_out

def minimise_R(atlas, frames, pos, frame_inds, pix_shifts):
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
        a[mask] = xx
        
        ax = poly_utils.polyder2d(a, ax, 0)
        ay = poly_utils.polyder2d(a, ay, 1)
        
        # make pixel_shifts
        R[0] = polyval2d(x, y, ax)
        R[1] = polyval2d(x, y, ay)
        print(R.max())
        
        f_out.fill(-1)
        for k in range(NN):
            ss = i - pos2[k][0] + R[0]
            fs = j - pos2[k][1] + R[1]
            ss = np.rint(ss).astype(np.int)
            fs = np.rint(fs).astype(np.int)
            m = (ss > 0) * (ss < atlas.shape[0]) * (fs > 0) * (fs < atlas.shape[1])
            
            f_out[k][m] =  atlas[ss[m], fs[m]]
        return R, f_out
    
    def fun(xx, ax, ay, a, R):
        print(xx)
        R, f_out = make_shifts_frames(xx, ax, ay, a, R) 
        out = 0
        for fr, im in zip(f_out, frames):
            t = feature_matching.similarity_pearson(fr, im, i0, j0)
            out -= t
        return out
    
    import time
    x0 = np.zeros(K**2 - 5, dtype=np.float)
    #x0 = np.random.random(x0.shape)

    d0 = time.time()
    res = scipy.optimize.minimize(fun, x0, args=(ax, ay, a, R), tol=1.0e-20, options={'disp' : True, 'eps' : 1.0e-2})
    d1 = time.time()
    print(res)
    print('time to optimise:', d1-d0, 's') 
    print('fun0   :', fun(x0, ax, ay, a, R))   
    print('fun out:', fun(res.x, ax, ay, a, R)) ; sys.stdout.flush()
    
    R, f_out = make_shifts_frames(res.x, ax, ay, a, R) 
    return R, f_out

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


if __name__ == '__main__':
    args, params = cmdline_parser.parse_cmdline_args('tracking_spline', 'parameterised refinement of pixel shifts')
    w   = params['window']
    roi = params['roi']
    g   = params['grid']
    
    f = h5py.File(args.filename, 'r')
    h5_params, fnam = config_reader.config_read_from_h5(args.config, f)
    h5_params       = h5_params['tracking_spline']
    frames          = [h5_params['data'][i, roi[0]:roi[1], roi[2]:roi[3]] for i in params['frames']]
    W               = h5_params['whitefield'][roi[0]:roi[1], roi[2]:roi[3]]
    mask            = h5_params['mask'][roi[0]:roi[1], roi[2]:roi[3]]
    pix_shifts      = h5_params['pixel_shifts'][:, roi[0]:roi[1], roi[2]:roi[3]]
    atlas           = h5_params['atlas'][:]
    pos             = h5_params['pix_positions'][:, 1:]
    f.close()
    
    W[W==0] = 1
    frames  = mask * np.array(frames).astype(np.float) / W.astype(np.float)

    
    #atlas_out = feature_matching.build_atlas_distortions(frames, W, pos[1:], pix_shifts[0].astype(np.int), pix_shifts[1].astype(np.int))
    #config_reader.write_h5(args.filename, params['h5_group'], 
    #                      {'atlas' : atlas_out})
    #print('display: '+params['h5_group']+'/atlas') ; sys.stdout.flush()
    
    # forward model a few frames
    #for_frames = forward_frames(atlas, pix_shifts, pos, params['frames'], frames.shape[1:])
    R, f_out = minimise_R(atlas, frames[27:31], pos, [28,29,30,31], pix_shifts)
    
    config_reader.write_h5(args.filename, params['h5_group'], 
                          {'frame_comp' : np.vstack((frames[27], f_out[0]))})
    print('display: '+params['h5_group']+'/frame_comp') ; sys.stdout.flush()

    config_reader.write_h5(args.filename, params['h5_group'], 
                          {'frames_forward' : f_out, 'R' : R})
    """
    index = 0
    for i in params['frames'][:20]:
        for_frame = forward_frames(atlas, pix_shifts, pos[1:], [params['frames'][i]-1,], frames.shape[1:])[0]
        config_reader.write_h5(args.filename, params['h5_group'], 
                              {'frame_comp' : np.vstack((frames[i], for_frame))})
        print('display: '+params['h5_group']+'/frame_comp') ; sys.stdout.flush()
        index += 1

        # calculate similarity
        im = frames[i]
        im[~mask] = -1
        ii, jj = -((1-im.shape[0])//2), -((1-im.shape[1])//2)
        print(feature_matching.similarity_pearson(for_frame, im, ii, jj))
        time.sleep(0.1)
    """

    """
    # make the ij grid within the roi:
    i = np.ogrid[w[0]//2 : roi[1]-roi[0]-w[0]//2 : g[0]*1J]
    j = np.ogrid[w[1]//2 : roi[3]-roi[2]-w[1]//2 : g[1]*1J]
    
    i, j    = np.meshgrid(i, j, indexing='ij')
    ij_grid = np.vstack((i.ravel(), j.ravel()))
    
    #config_reader.write_h5(args.filename, params['h5_group'], 
    #                      {'frames_forward' : for_frames})
    #print('display:', params['atlas']) ; sys.stdout.flush()
    """
    
    # anoying but need to do this to give the 
    # widget time to display
    time.sleep(1)
