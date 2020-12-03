cimport numpy as np
import numpy as np
import tqdm
from libc.math cimport ceil, floor, sqrt, exp, pi
from cython.parallel import prange
cimport openmp
from . import utils_opencl

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.npy_bool bool_t

DEF FLOAT_MAX = 1.7976931348623157e+308
DEF MU_C = 1.681792830507429
DEF NO_VAR = -1.0

cdef float_t rbf(float_t dsq, float_t ls) nogil:
    return exp(-dsq / 2 / ls**2) / sqrt(2 * pi)

cdef void mse_bi(float_t* m_ptr, float_t[::1] I, float_t[:, ::1] I0,
                 float_t[:, ::1] dij_n, float_t ux, float_t uy) nogil:
    cdef:
        int a = I.shape[0] - 1, aa = I0.shape[0], bb = I0.shape[1]
        int i, ss0, ss1, fs0, fs1
        float_t SS_res = 0, SS_tot = 0, ss, fs, dss, dfs, I0_bi
    for i in range(a):
        ss = ux - dij_n[i, 0]
        fs = uy - dij_n[i, 1]
        if ss <= 0:
            dss = 0; ss0 = 0; ss1 = 0
        elif ss >= aa - 1:
            dss = 0; ss0 = aa - 1; ss1 = aa - 1
        else:
            ss = ss; dss = ss - floor(ss)
            ss0 = <int>(floor(ss)); ss1 = ss0 + 1
        if fs <= 0:
            dfs = 0; fs0 = 0; fs1 = 0
        elif fs >= bb - 1:
            dfs = 0; fs0 = bb - 1; fs1 = bb - 1
        else:
            fs = fs; dfs = fs - floor(fs)
            fs0 = <int>(floor(fs)); fs1 = fs0 + 1
        I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                (1 - dss) * dfs * I0[ss0, fs1] + \
                dss * (1 - dfs) * I0[ss1, fs0] + \
                dss * dfs * I0[ss1, fs1]
        SS_res += (I[i] - I0_bi)**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res / SS_tot
    if m_ptr[1] >= 0:
        m_ptr[1] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

cdef void krig_data_c(float_t[::1] I, float_t[:, :, ::1] I_n, bool_t[:, ::1] M, float_t[:, ::1] W,
                      float_t[:, :, ::1] u, int j, int k, float_t ls) nogil:
    cdef:
        int a = I_n.shape[0], b = I_n.shape[1], c = I_n.shape[2], i, jj, kk
        int djk = <int>(ceil(2 * ls))
        int jj0 = j - djk if j - djk > 0 else 0
        int jj1 = j + djk if j + djk < b else b
        int kk0 = k - djk if k - djk > 0 else 0
        int kk1 = k + djk if k + djk < c else c
        float_t w0 = 0, rss = 0, r
    for i in range(a + 1):
        I[i] = 0
    for jj in range(jj0, jj1):
        for kk in range(kk0, kk1):
            r = rbf((u[0, jj, kk] - u[0, j, k])**2 + (u[1, jj, kk] - u[1, j, k])**2, ls)
            w0 += r * M[jj, kk] * W[jj, kk]**2
            rss += M[jj, kk] * W[jj, kk]**3 * r**2
            for i in range(a):
                I[i] += I_n[i, jj, kk] * M[jj, kk] * W[jj, kk] * r
    if w0:
        for i in range(a):
            I[i] /= w0
        I[a] = rss / w0**2

def calc_error_total(float_t[:, :, ::1] data, bool_t[:, ::1] mask, float_t[:, ::1] W, float_t[:, ::1] dij_n,
                     float_t[:, ::1] O, float_t[:, :, ::1] pixel_map, float_t n0, float_t m0, float_t ls, roi=None):
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = data.shape[0], b = data.shape[1], c = data.shape[2]
        int aa = O.shape[0], bb = O.shape[1], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t err = 0
        float_t[:, ::1] mptr = NO_VAR * np.ones((max_threads, 2), dtype=dtype)
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
    for k in prange(b, schedule='static', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(c):
            krig_data_c(I[t], data, mask, W, pixel_map, j, k, ls)
            mse_bi(&mptr[t, 0], I[t], O, dij_n, pixel_map[0, j, k], pixel_map[1, j, k])
            err += mptr[t, 0]
    return err / b / c

def calc_error(data, mask, W, dij_n, O, pixel_map, n0, m0, ls,
               subpixel=True, verbose=True):
    r"""
    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L).
    
    mask : ndarray
        Boolean array of shape (M, L), where True indicates a good
        pixel and False a bad pixel.
    
    W : ndarray
        Float array of shape (M, L) containing the estimated whitefield.
    
    dij_n : ndarray
        Float array of shape (N, 2) containing the object translations 
        that have been mapped onto the detector's frame of reference.     

    O : ndarray
        Float array of shape (U, V), this is essentially an object map. 
    
    pixel_map : ndarray, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]
    
    n0 : float
        Slow scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \ge -0.5 
            \quad\text{for all } i,j
    
    m0 : float
        Fast scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0 \ge -0.5 
            \quad\text{for all } i,j
    
    verbose : bool, optional
        print what I'm doing.
    
    minimum_overlap : float or None, optional
        Default is None. If float then the the object will be set to -1 
        where the number of data points contributing to that value is less
        than "minimum_overlap".

    Returns
    -------
    
    error_total : float
        The global error value, :math:`\varepsilon = \sum_{n,i,j} \varepsilon[n, i, j]`.
    
    Notes
    -----
    The error, per pixel and per frame, is given by:

    .. math::

        \begin{align}
        \varepsilon[n, i, j] = M[i,j] \bigg[ I_\Phi[n, i, j] - 
            W[i, j] I_0[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                        &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg]^2
        \end{align}
    """
    # mask the pixel mapping
    ij          = np.array([pixel_map[0], pixel_map[1]])
    norm        = np.zeros(data.shape[1:])
    flux_corr   = np.zeros(data.shape[0])

    # for a 1d dataset add the ptychograph for data vs. forward
    data_1d = False
    if 1 in data.shape :
        data_1d = True
    forward = -np.ones(data.shape, dtype=np.float32)
    
    #sig = np.std(data, axis=0)
    #sig[sig <= 0] = 1
    for n in tqdm.trange(data.shape[0], desc='calculating errors'):
        # define the coordinate mapping 
        ss = pixel_map[0] - dij_n[n, 0] + n0
        fs = pixel_map[1] - dij_n[n, 1] + m0
        
        if subpixel: 
            #I0 = W * bilinear_interpolation_array(I, ss, fs, fill=-1, invalid=-1)
            #I0 = I0[mask]
            I0 = W * utils_opencl.bilinear_interpolation_array(O, ss, fs)
        
        else :
            # round to int
            I0 = O[np.rint(ss).astype(np.int), np.rint(ss).astype(np.int)] * W
        
        d  = data[n]
        m  = (I0>0)*(d>0)*mask
        norm += m*(W - d)**2
                
        # flux correction factor
        flux_corr[n] = np.sum(m * I0 * data[n]) / np.sum(m * data[n]**2) 

        forward[n] = I0
    
    norm /= data.shape[0]
    norm[norm==0] = 1
    
    error_total = calc_error_total(data, mask, W, dij_n, O, pixel_map, n0, m0, ls)
    
    if data_1d :
        res = {'1d_data_vs_forward': np.squeeze(np.array([data, forward]))}
    else :
        res = {'forward': forward}
    
    return error_total, norm, flux_corr, res