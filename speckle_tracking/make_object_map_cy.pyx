cimport numpy as np
import numpy as np
from libc.math cimport ceil, sqrt, exp, pi
from cython.parallel import prange
cimport openmp

ctypedef fused float_t:
    np.float64_t
    np.float32_t

ctypedef np.npy_bool bool_t

cdef float_t rbf(float_t dsq, float_t ls) nogil:
    return exp(-dsq / 2 / ls**2) / sqrt(2 * pi)

cdef void frame_reference(float_t[:, ::1] I0, float_t[:, ::1] w0, float_t[:, ::1] I, bool_t[:, ::1] M,
                          float_t[:, ::1] W, float_t[:, :, ::1] u, float_t di, float_t dj, float_t ls) nogil:
    cdef:
        int b = I.shape[0], c = I.shape[1], j, k, jj, kk, j0, k0
        int aa = I0.shape[0], bb = I0.shape[1], jj0, jj1, kk0, kk1
        int dn = <int>(ceil(4 * ls))
        float_t ss, fs, r
    for j in range(b):
        for k in range(c):
            ss = u[0, j, k] - di
            fs = u[1, j, k] - dj
            j0 = <int>(ss) + 1
            k0 = <int>(fs) + 1
            jj0 = j0 - dn if j0 - dn > 0 else 0
            jj1 = j0 + dn if j0 + dn < aa else aa
            kk0 = k0 - dn if k0 - dn > 0 else 0
            kk1 = k0 + dn if k0 + dn < bb else bb
            for jj in range(jj0, jj1):
                for kk in range(kk0, kk1):
                    r = rbf((jj - ss)**2 + (kk - fs)**2, ls)
                    I0[jj, kk] += I[j, k] * M[j, k] * W[j, k] * r
                    w0[jj, kk] += M[j, k] * W[j, k]**2 * r

def make_object_map(float_t[:, :, ::1] data, bool_t[:, ::1] mask, float_t[:, ::1] W, float_t[:, ::1] dij_n,
                    float_t[:, :, ::1] pixel_map, float_t ls):
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
    
    pixel_map : ndarray, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]

    ls : float
        Object map length scale in pixels.

    Returns
    -------
    I : ndarray
        Float array of shape (U, V), this is essentially an object map given by:
        
        .. math::

            I_0[i, j] &= 
            \frac{\sum_n M[u_n, v_n] W[u_n, v_n] I^{z_1}_\phi[n, u_n, v_n]}{\sum_n M[u_n, v_n] W[u_n, v_n]^2 } \\
        
        where: 
        
        .. math::

            \begin{align}
            u_n[i,j] &= \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \\
            u_n[i,j] &= \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \\
            \end{align}
        
        see Notes for more.
    
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
        
        -0.5 is chosen rather than 0 because integer coordinates are defined 
        at the centre of the physical pixel locations.
    
    Notes
    -----
    .. math::
        
        I_{\phi, n}(\mathbf{x})
        = W(\mathbf{x})I_0(\mathbf{x} - \frac{\lambda z}{2\pi} \nabla \Phi(x)-\Delta x_n, \bar{z}_\Phi)
    
    :math:`M, W` are the mask and whitefield arrays respectively. 
    U and V are the pixel dimensions of :math:`I_0` given by:
     
    .. math::
        
        U &= \text{max}(\text{ij}_\text{map}[0, i, j]) - \text{min}(\Delta ij_n[0]) + n_0 \\
        V &= \text{max}(\text{ij}_\text{map}[1, i, j]) - \text{min}(\Delta ij_n[1]) + m_0
    """
    dtype = np.float64 if float_t is np.float64_t else np.float32
    cdef:
        int a = data.shape[0], b = data.shape[1], c = data.shape[2], i, j, k, t
        float_t n0 = -np.min(pixel_map[0]) + np.max(dij_n[:, 0])
        float_t m0 = -np.min(pixel_map[1]) + np.max(dij_n[:, 1])
        int aa = <int>(np.max(pixel_map[0]) - np.min(dij_n[:, 0]) + n0) + 1
        int bb = <int>(np.max(pixel_map[1]) - np.min(dij_n[:, 1]) + m0) + 1
        int max_threads = openmp.omp_get_max_threads()
        float_t[:, :, ::1] I = np.zeros((max_threads, aa, bb), dtype=dtype)
        float_t[:, :, ::1] w = np.zeros((max_threads, aa, bb), dtype=dtype)
        float_t[::1] Is = np.empty(max_threads, dtype=dtype)
        float_t[::1] ws = np.empty(max_threads, dtype=dtype)
        float_t[:, ::1] I0 = np.zeros((aa, bb), dtype=dtype)
    for i in prange(a, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        frame_reference(I[t], w[t], data[i], mask, W, pixel_map, dij_n[i, 0] - n0, dij_n[i, 1] - m0, ls)
    for k in prange(bb, schedule='guided', nogil=True):
        t = openmp.omp_get_thread_num()
        for j in range(aa):
            Is[t] = 0; ws[t] = 0
            for i in range(max_threads):
                Is[t] = Is[t] + I[i, j, k]
                ws[t] = ws[t] + w[i, j, k]
            if ws[t]:
                I0[j, k] = Is[t] / ws[t]
            else:
                I0[j, k] = 0
    return np.asarray(I0), n0, m0
