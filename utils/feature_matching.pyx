import numpy as np
cimport numpy as np

from libc.math cimport sin, cos, acos, exp, sqrt, fabs, M_PI, ceil, floor

FLOAT = np.float
INT   = np.int

ctypedef np.float_t FLOAT_t
ctypedef np.int_t INT_t

cimport cython

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def similarity_lstsq(np.ndarray[FLOAT_t, ndim=2] fe, np.ndarray[FLOAT_t, ndim=2] im, int i, int j):
    """
    """
    cdef int offset_i, offset_j, ii, jj, iii, jjj
    cdef float err, tot
    offset_i = i+(1-fe.shape[0])//2
    offset_j = j+(1-fe.shape[1])//2
    
    err = 0
    tot = 0
    for ii in range(fe.shape[0]):
        for jj in range(fe.shape[1]):
            iii = ii + offset_i
            jjj = jj + offset_j
                
            # bounds checking
            if iii >= 0 and jjj >= 0 :
                if iii < im.shape[0] and jjj < im.shape[1] :
                    if fe[ii, jj] > 0 and im[iii, jjj] > 0 :
                        err += (fe[ii, jj] - im[iii, jjj])**2
                        tot += im[iii, jjj]**2
    return err/tot

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def similarity_pearson(np.ndarray[FLOAT_t, ndim=2] fe, np.ndarray[FLOAT_t, ndim=2] im, int i, int j):
    """
    """
    cdef int offset_i, offset_j, ii, jj, iii, jjj
    cdef float err = 0, tot = 0, fe_mean = 0, im_mean = 0, \
               fe2_mean = 0, im2_mean = 0, fe_im_mean = 0, count = 0
    offset_i = i+(1-fe.shape[0])//2
    offset_j = j+(1-fe.shape[1])//2
    
    for ii in range(fe.shape[0]):
        for jj in range(fe.shape[1]):
            iii = ii + offset_i
            jjj = jj + offset_j
                
            # bounds checking
            if iii >= 0 and jjj >= 0 :
                if iii < im.shape[0] and jjj < im.shape[1] :
                    if fe[ii, jj] > 0 and im[iii, jjj] > 0 :
                        fe_mean    += fe[ii, jj]
                        im_mean    += im[iii, jjj]
                        fe2_mean   += fe[ii, jj]**2
                        im2_mean   += im[iii, jjj]**2
                        fe_im_mean += fe[ii, jj] * im[iii, jjj]
                        count      += 1
    tot = (count*fe_im_mean - fe_mean*im_mean)
    tot = tot / sqrt((count*fe2_mean - fe_mean**2)*(count*im2_mean - im_mean**2))
    return tot

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def feature_err_map(np.ndarray[FLOAT_t, ndim=2] feature, np.ndarray[FLOAT_t, ndim=2] image):
    cdef np.ndarray[FLOAT_t, ndim=2] err = np.zeros((image.shape[0], image.shape[1]), dtype=FLOAT)
    cdef int i, j
    for i in range(image.shape[0]):
        for j in range(image.shape[1]):
            err[i, j] = similarity_pearson(feature, image, i, j)
    return err

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def feature_err_map_range(np.ndarray[FLOAT_t, ndim=2] feature, np.ndarray[FLOAT_t, ndim=2] image, \
                          int i_min,  int i_max,  int j_min,  int j_max):
    cdef np.ndarray[FLOAT_t, ndim=2] err = np.zeros((image.shape[0], image.shape[1]), dtype=FLOAT)
    cdef int i, j
    for i in range(i_min, i_max+1, 1):
        for j in range(j_min, j_max+1, 1):
            err[i, j] = -similarity_pearson(feature, image, i, j)
    return err

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def err_dx(errs, ij_grid, sig_u, range_u, dis, djs):
    """
    return 
        e_ij(dx)    = min_u [e_ij(dx, u)]
    where: 
    e_ij(dx, u) = sum_k [-Pearson(x_k, x_k + u(x_k) + dx_ij) + u(x_k)**2 / sig_u**2]
    and:
    x_k are the centre of the speckles, u(x_k) is the distortion term
    """
    out = np.zeros((len(dis), len(djs)), dtype=np.float)
    for ii, dx in enumerate(dis):
        for jj, dy in enumerate(djs):
            ui     = np.arange(-range_u, range_u, 1)
            ks     = np.arange(len(errs))
            
            ks, ui, uj = np.meshgrid(ks, ui, ui, indexing='ij')
            
            i = ij_grid[ks, np.zeros_like(ks)] + dx + ui
            j = ij_grid[ks, np.ones_like(ks) ] + dy + uj
            
            m = (i>=0) * (i<errs.shape[1]) * (j>=0) * (j<errs.shape[2])
            i[~m] = 0
            j[~m] = 0
            
            temp = errs[ks, i, j]
            temp += (ui**2+uj**2)/sig_u**2
            temp[~m] = 0
            
            err = np.sum(np.min(temp, axis=(1,2)))
            
            out[ii, jj] = err
    return out

def find_shift(frame0, frame1, w, linear_overlap, N_features, sig_u):
    # make a grid of points, avoiding the edges
    N = N_features
    ij_grid = []
    for i in np.linspace(0, frame1.shape[0], N+2):
        for j in np.linspace(0, frame1.shape[1], N+2):
            if i != 0 and i != frame1.shape[0]:
                if j != 0 and j != frame1.shape[1]:
                    ij_grid.append( (int(round(i)), int(round(j))) )

    # extract the features to match
    features = []
    for i, j in ij_grid:
        i_min, i_max = max(i+(1-w[0])//2, 0), min(i+(w[0]+1)//2, frame0.shape[0])
        j_min, j_max = max(j+(1-w[1])//2, 0), min(j+(w[1]+1)//2, frame0.shape[1])
        features.append(frame0[i_min:i_max, j_min:j_max])
    
    # calculate the feature correlation maps
    feature_fit_maps = []
    for feature in features :
        feature_fit_maps.append(feature_err_map(feature, frame1))

    # calculate the error corresponding to each step 
    dis = np.arange( -int(round((1-linear_overlap) * frame0.shape[0])), 
                      int(round((1-linear_overlap) * frame0.shape[0])), 1)
    djs = np.arange( -int(round((1-linear_overlap) * frame0.shape[1])), 
                      int(round((1-linear_overlap) * frame0.shape[1])), 1)
    
    # evaluate the error for each relative shift
    errors_dx = err_dx(np.array(feature_fit_maps),
                                np.array(ij_grid), sig_u, 
                                int(round(3*sig_u)), dis, djs)
    
    # choose the relative shift of least error
    i, j = np.unravel_index(np.argmin(errors_dx), errors_dx.shape)

    return [dis[i],djs[j]], errors_dx[i, j]

def build_atlas(frames, steps):
    """
    given I_i(x) = O(x-x_i)

    estimate O(x) = sum_i I_i(x+x_i) / norm
    """
    steps2 = np.rint(np.array(steps)-np.max(steps, axis=0)).astype(np.int)
    
    # define the atlas grid
    shape = frames[0].shape
    N, M  = (shape[0] - np.min(steps2[:, 0]), shape[1] - np.min(steps2[:, 1]))
    atlas   = np.zeros((N, M), dtype=np.float)
    overlap = np.zeros((N, M), dtype=np.int)
    i, j = np.ogrid[0:shape[0], 0:shape[1]]

    for frame, step in zip(frames, steps2):
        atlas[i-step[0]  , j-step[1]] += (frame>0)*frame
        overlap[i-step[0], j-step[1]] += (frame>0)
    overlap[overlap==0] = 1
    return atlas / overlap.astype(np.float)

def build_atlas_distortions(frames, W, steps, uss, ufs):
    # the regular pixel values
    shape = frames[0].shape
    i, j  = np.ogrid[0:shape[0], 0:shape[1]]
    
    # define the atlas grid
    steps2 = np.array(steps)
    steps2[:, 0] += -np.max(steps[:, 0])+np.min(i + uss)
    steps2[:, 1] += -np.max(steps[:, 1])+np.min(j + ufs)
    steps2 = np.rint(steps2).astype(np.int)

    uss2 = np.rint(uss).astype(np.int)
    ufs2 = np.rint(ufs).astype(np.int)
    
    mask0 = frames[0]>0
    N, M  = (- np.min(steps2[:, 0]) + np.max(i + uss), np.max(j + ufs) - np.min(steps2[:, 1]))
    N = int(round(N))
    M = int(round(M))
    WW = W**2 
    
    atlas   = np.zeros((N, M), dtype=np.float)
    overlap = np.zeros((N, M), dtype=np.float)
    
    for frame, step in zip(frames, steps2):
        ss = i + uss2 - step[0]
        fs = j + ufs2 - step[1]
        mask = (ss > 0) * (ss < N) * (fs > 0) * (fs < M) * mask0
        atlas[  ss[mask], fs[mask]] += (frame*W)[mask]
        overlap[ss[mask], fs[mask]] += WW[mask] 
    atlas /= (overlap + 1.0e-5)
    return atlas

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def sub_pixel_warp(np.ndarray[FLOAT_t, ndim=2] atlas, np.ndarray[FLOAT_t, ndim=2] out, np.ndarray[FLOAT_t, ndim=3] pixel_shifts, float pos_ss, float pos_fs):
    """
    apply linear sub-pixel interpolation to get the function:
        out[i, j] = atlas[i + pixel_shifts[0] - pos_ss, j + pixel_shifts[1] - pos_fs]
    
    for float interpolation we use: https://en.wikipedia.org/wiki/Bilinear_interpolation#Unit_square
    """
    cdef int i, j, i0, i1, j0, j1
    cdef float x, y
    for i in range(out.shape[0]):
        for j in range(out.shape[1]):
            x  = i + pixel_shifts[0, i, j] - pos_ss
            y  = j + pixel_shifts[1, i, j] - pos_fs
            if x < 0 or x > (atlas.shape[0]-1) or y < 0 or y > (atlas.shape[1]-1):
                out[i, j] = -1
            else :
                i0, i1 = int(floor(x)), int(ceil(x))
                j0, j1 = int(floor(y)), int(ceil(y))
                if i1 == i0 :
                    i1 += 1
                if j1 == j0 :
                    j1 += 1
                out[i, j] = atlas[i0, j0] * (i1-x) * (j1-y) + atlas[i1, j0] * (x-i0) * (j1-y) \
                          + atlas[i0, j1] * (i1-x) * (y-j0) + atlas[i1, j1] * (x-i0) * (y-j0)


@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False)  # turn off negative index wrapping for entire function
def build_atlas_warp(np.ndarray[FLOAT_t, ndim=3] frames, np.ndarray[FLOAT_t, ndim=2] W, np.ndarray[FLOAT_t, ndim=3] pixel_shifts, np.ndarray[FLOAT_t, ndim=2] steps):
    """
    """
    cdef int i, j, k, i0, i1, j0, j1, N, M
    cdef float x, y, t, w, ww, n, m
    
    # the regular pixel values
    shape = frames[0].shape
    ii, jj  = np.ogrid[0:shape[0], 0:shape[1]]
    
    # offset the steps
    steps2 = np.array(steps)
    steps2[:, 0] += -np.max(steps[:, 0])+np.min(ii + pixel_shifts[0])
    steps2[:, 1] += -np.max(steps[:, 1])+np.min(jj + pixel_shifts[1])
    
    # define the atlas grid
    n, m  = (- np.min(steps2[:, 0]) + np.max(ii + pixel_shifts[0]), np.max(jj + pixel_shifts[1]) - np.min(steps2[:, 1]))
    N = int(ceil(n))
    M = int(ceil(m))
    WW = W**2
    atlas   = -np.ones((N, M), dtype=np.float)
    overlap = -np.ones((N, M), dtype=np.float)

    for k in range(frames.shape[0]):
        print(k)
        for i in range(frames.shape[1]):
            for j in range(frames.shape[2]):
                x  = i + pixel_shifts[0, i, j] - steps2[k][0]
                y  = j + pixel_shifts[1, i, j] - steps2[k][1]
                if not (x < 0 or x > (atlas.shape[0]-1) or y < 0 or y > (atlas.shape[1]-1)):
                    t  = frames[k, i, j]
                    w  = W[i, j]
                    ww = WW[i, j]
                    if t > 0 and w > 0 :
                        i0, i1 = int(floor(x)), int(ceil(x))
                        j0, j1 = int(floor(y)), int(ceil(y))
                        if i1 == i0 :
                            if i1 == atlas.shape[0] :
                                i0 -= 1
                            else :
                                i1 += 1
                        if j1 == j0 :
                            if j1 == atlas.shape[1] :
                                j0 -= 1
                            else :
                                j1 += 1
                        
                        r0 = (i1-x) * (j1-y)
                        r1 = (x-i0) * (j1-y)
                        r2 = (i1-x) * (y-j0)
                        r3 = (x-i0) * (y-j0)
                        atlas[i0, j0] += t * r0 * w
                        atlas[i1, j0] += t * r1 * w
                        atlas[i0, j1] += t * r2 * w
                        atlas[i1, j1] += t * r3 * w
                        
                        overlap[i0, j0] += r0 * ww
                        overlap[i1, j0] += r1 * ww
                        overlap[i0, j1] += r2 * ww
                        overlap[i1, j1] += r3 * ww
        
    atlas[atlas>0] = atlas[atlas>0] / (overlap[atlas>0]+1.0e-5)
    return atlas
