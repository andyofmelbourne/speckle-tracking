cimport numpy as np
import numpy as np
from libc.math cimport ceil, floor, sqrt, exp, pi
from scipy.ndimage import gaussian_filter
from cython.parallel import prange
cimport openmp

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.npy_bool bool_t

DEF FLOAT_MAX = 1.7976931348623157e+308
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
            dss = ss - floor(ss)
            ss0 = <int>(floor(ss)); ss1 = ss0 + 1
        if fs <= 0:
            dfs = 0; fs0 = 0; fs1 = 0
        elif fs >= bb - 1:
            dfs = 0; fs0 = bb - 1; fs1 = bb - 1
        else:
            dfs = fs - floor(fs)
            fs0 = <int>(floor(fs)); fs1 = fs0 + 1
        I0_bi = (1 - dss) * (1 - dfs) * I0[ss0, fs0] + \
                (1 - dss) * dfs * I0[ss0, fs1] + \
                dss * (1 - dfs) * I0[ss1, fs0] + \
                dss * dfs * I0[ss1, fs1]
        SS_res += (I[i] - I0_bi)**2
        SS_tot += (I[i] - 1)**2
    m_ptr[0] = SS_res; m_ptr[1] = SS_tot
    if m_ptr[2] >= 0:
        m_ptr[2] = 4 * I[a] * (SS_res / SS_tot**2 + SS_res**2 / SS_tot**3)

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

cdef void subpixel_ref_1d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
    cdef:
        float_t dfs = 0, det, dd
    det = 4 * (mse_m[2] + mse_m[0] - 2 * mse_m[1])
    if det != 0:
        dfs = (mse_m[0] - mse_m[2]) / det * mu
        dd = sqrt(dfs**2)
        if dd > 1:
            dfs /= dd

    x[1] += dfs

cdef void subpixel_ref_2d(float_t[::1] x, float_t* mse_m, float_t mu) nogil:
    cdef:
        float_t dss = 0, dfs = 0, det, dd
    det = 4 * (mse_m[5] + mse_m[1] - 2 * mse_m[3]) * (mse_m[4] + mse_m[2] - 2 * mse_m[3]) - \
          (mse_m[6] + mse_m[0] + 2 * mse_m[3] - mse_m[1] - mse_m[5] - mse_m[2] - mse_m[4])**2
    if det != 0:
        dss = ((mse_m[6] + mse_m[0] + 2 * mse_m[3] - mse_m[1] - mse_m[5] - mse_m[2] - mse_m[4]) * \
               (mse_m[4] - mse_m[2]) - 2 * (mse_m[4] + mse_m[2] - 2 * mse_m[3]) * \
               (mse_m[5] - mse_m[1])) / det * mu / 2
        dfs = ((mse_m[6] + mse_m[0] + 2 * mse_m[3] - mse_m[1] - mse_m[5] - mse_m[2] - mse_m[4]) * \
               (mse_m[5] - mse_m[1]) - 2 * (mse_m[5] + mse_m[1] - 2 * mse_m[3]) * \
               (mse_m[4] - mse_m[2])) / det * mu / 2
        dd = sqrt(dfs**2 + dss**2)
        if dd > 1:
            dss /= dd; dfs /= dd
    
    x[0] += dss; x[1] += dfs

cdef void update_pm_c(float_t[::1] I, float_t[:, ::1] I0, float_t[::1] u, float_t[:, ::1] dij_n,
                      float_t n0, float_t m0, int sw_ss, int sw_fs) nogil:
    cdef:
        int ss_min = -sw_ss, fs_min = -sw_fs, ss_max = -sw_ss, fs_max = -sw_fs, ss, fs
        float_t mse_min = FLOAT_MAX, mse_max = -FLOAT_MAX, mse_var = FLOAT_MAX, mse, l1, mu
        float_t mv_ptr[3]
        float_t mse_m[7]
    for ss in range(-sw_ss, sw_ss + 1):
        for fs in range(-sw_fs, sw_fs + 1):
            mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0 + ss , u[1] + m0 + fs)
            mse = mv_ptr[0] / mv_ptr[1]
            if mse < mse_min:
                mse_min = mse; mse_var = mv_ptr[2]; ss_min = ss; fs_min = fs; 
            if mse > mse_max:
                mse_max = mse; ss_max = ss; fs_max = fs
    u[0] += ss_min; u[1] += fs_min
    l1 = 2 * (mse_max - mse_min) / ((ss_max - ss_min)**2 + (fs_max - fs_min)**2)
    mu = (3 * mse_var**0.5 / l1)**0.33
    mu = mu if mu > 2 else 2
    if sw_ss:
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0 - mu / 2, u[1] + m0 - mu / 2)
        mse_m[0] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0 - mu / 2, u[1] + m0)
        mse_m[1] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0, u[1] + m0 - mu / 2)
        mse_m[2] = mv_ptr[0] / mv_ptr[1]
        mse_m[3] = mse_min
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0, u[1] + m0 + mu / 2)
        mse_m[4] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0 + mu / 2, u[1] + m0)
        mse_m[5] = mv_ptr[0] / mv_ptr[1]
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0 + mu / 2, u[1] + m0 + mu / 2)
        mse_m[6] = mv_ptr[0] / mv_ptr[1]
        subpixel_ref_2d(u, mse_m, mu)
    else:
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0, u[1] + m0 - mu / 2)
        mse_m[0] = mv_ptr[0] / mv_ptr[1]
        mse_m[1] = mse_min
        mse_bi(mv_ptr, I, I0, dij_n, u[0] + n0, u[1] + m0 + mu / 2)
        mse_m[2] = mv_ptr[0] / mv_ptr[1]
        subpixel_ref_1d(u, mse_m, mu)

def update_pixel_map(float_t[:, :, ::1] data, bool_t[:, ::1] mask, float_t[:, ::1] W, float_t[:, ::1] dij_n,
                     float_t[:, ::1] O, float_t[:, :, ::1] pixel_map, float_t n0, float_t m0,
                     int sw_ss, int sw_fs, float_t ls):
    r"""
    Update the pixel_map by minimising an error metric within the search window of size [sw_ss, sw_fs].
    
    Parameters
    ----------
    data : ndarray, float32, (N, M, L)
        Input diffraction data :math:`I^{z_1}_\phi`, the :math:`^{z_1}` 
        indicates the distance between the virtual source of light and the 
        :math:`_\phi` indicates the phase of the wavefront incident on the 
        sample surface. The dimensions are given by:
        
        - N = number of frames
        - M = number of pixels along the slow scan axis of the detector
        - L = number of pixels along the fast scan axis of the detector
    
    mask : ndarray, bool, (M, L)
        Detector good / bad pixel mask :math:`M`, where True indicates a good
        pixel and False a bad pixel.
    
    W : ndarray, float, (M, L)
        The whitefield image :math:`W`. This is the image one obtains without a 
        sample in place.

    dij_n : ndarray, float, (N, 2)
        An array containing the sample shifts for each detector image in pixel units
        :math:`\Delta ij_n`.

    O : ndarray
        Float array of shape (U, V), this is essentially an object map. 

    pixel_map : ndarray, float, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object :math:`ij_\text{map}`, such that: 
        
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

    sw_ss : int
        The pixel mapping will be updated in a square area of side length "sw_ss" along the detector's slow axis.
        This value/s are in pixel units.

    sw_fs : int
        The pixel mapping will be updated in a square area of side length "sw_ss" along the detector's fast axis.
        This value/s are in pixel units.

    ls : float
        Pixel mapping length scale in pixel units.
    
    Returns
    -------
    pixel_map : ndarray, float, (2, M, L)
        An array containing the updated pixel mapping.

    Notes
    -----
    The following error metric is minimised with respect to :math:`\text{ij}_\text{map}[0, i, j]`:

    .. math:: 
    
        \begin{align}
        \varepsilon[i, j] = 
            \sum_n \bigg(I^{z_1}_{\phi}[n, i, j]
            - W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg)^2
        \end{align}
    """
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = data.shape[0], b = data.shape[1], c = data.shape[2], j, k, t
        int max_threads = openmp.omp_get_max_threads()
        float_t[::1, :, :] pixel_map_r = np.empty((2, b, c), dtype=dtype, order='F')
        float_t[:, ::1] I = np.empty((max_threads, a + 1), dtype=dtype)
    for k in prange(c, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(b):
            krig_data_c(I[t], data, mask, W, pixel_map, j, k, ls)
            pixel_map_r[:, j, k] = pixel_map[:, j, k]
            update_pm_c(I[t], O, pixel_map_r[:, j, k], dij_n, n0, m0, sw_ss, sw_fs)
    return np.asarray(pixel_map_r, order='C')