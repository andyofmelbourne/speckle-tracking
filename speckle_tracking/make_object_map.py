import numpy as np

def make_object_map(data, mask, W, dij_n, pixel_map_inv):
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
    
    pixel_map_inv : ndarray
        Array of shape (2,) + shape containing the pixel mapping 
        between a detector frame and the object, such that: 
        
        .. math:: 
        
            I^\infty[i- \Delta ij_n[0] + n_0, j - \Delta ij_n[1] + m_0] = 
            I^{z_1}_{\phi}[\text{ij}^{-1}_\text{map}[0, i, j], \text{ij}^{-1}_\text{map}[1, i, j]]
    
    Returns
    -------
    I : ndarray
        Float array of shape (U, V), this is essentially an object map given by:
        
        .. math::

            I^\infty[i, j] &= 
            \frac{\sum_n M[u_n, v_n] W[u_n, v_n] I^{z_1}_\phi[u_n, v_n]}{\sum_n M[u_n, v_n] W[u_n, v_n]^2 } \\
            \text{where } (u_n, v_n) &= \left( i + \Delta ij_n[0] - n_0, j + \Delta ij_n[1] - m_0\right)
        
        see Notes for more.
    
    Notes
    -----
    .. math::
        
        I^\infty(x-\Delta x_n, z_\text{eff}) = A^{-2}(x) f_n(x - \frac{\lambda z_\text{eff}}{2\pi} \nabla \Phi(x))

    :math:`M, W` are the mask and whitefield arrays respectively. 
    U and V are given by:
     
    .. math::
        
        U &= \text{max}(\text{pixel_map_inv}[0]) + \text{max}(\Delta ij_n[0]) - \text{min}(\Delta ij_n[0]) \\
        V &= \text{max}(\text{pixel_map_inv}[1]) + \text{max}(\Delta ij_n[1]) - \text{min}(\Delta ij_n[1]) 

    n0, m0 are given by:
    
    .. math::
        
        (n_0, m_0) = \left(\text{max}(\Delta ij_n[0]), \text{max}(\Delta ij_n[1]) \right)
    """
    # round dij_n and pixel_map_inv to integers
    dij_nr = np.rint(dij_n).astype(np.int)
    ij     = np.rint(pixel_map_inv).astype(np.int)
    
    # choose the offset so that i - dij_n + n0 > 0
    nm0 = np.max(dij_nr, axis=0)
    
    i, j = np.indices(pixel_map_inv.shape[1:])
    
    # determin the object-map domain
    shape   = (i.shape[0] + np.max(nm0[0]-dij_nr[:, 0]), i.shape[1] + np.max(nm0[1]-dij_nr[:, 1]))
    I       = np.zeros(shape, dtype=np.float)
    overlap = np.zeros(shape, dtype=np.float)
    
    for n in range(data.shape[0]):
        I[      i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += (mask*W*data[n])[ij[0], ij[1]]
        overlap[i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += (mask*W**2     )[ij[0], ij[1]]
    
    overlap[overlap<1e-2] = -1
    m = (overlap > 0)
    
    I[m]  = I[m] / overlap[m]
    I[~m] = -1
    return I
