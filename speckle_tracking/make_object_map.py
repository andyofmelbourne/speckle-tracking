import numpy as np

def make_object_map(data, mask, W, dij_n, pixel_map):
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
    
    Returns
    -------
    I : ndarray
        Float array of shape (U, V), this is essentially an object map given by:
        
        .. math::

            I^\infty[i, j] &= 
            \frac{\sum_n M[u_n, v_n] W[u_n, v_n] I^{z_1}_\phi[n, u_n, v_n]}{\sum_n M[u_n, v_n] W[u_n, v_n]^2 } \\
        
        see Notes for more.
    
    n0 : float
        Slow scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \ge 0 
            \quad\text{for all} i,j

    m0 : float
        Fast scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0 \ge 0 
            \quad\text{for all} i,j

    Notes
    -----
    .. math::
        
        I^{z_1}_{\phi, n}(\mathbf{x})
        = W(\mathbf{x})I^\infty(x- \frac{\lambda \mathbf{z}_\text{eff}}{2\pi} \circ \nabla \phi(x)-\Delta x_n, z_\text{eff})
    
    :math:`M, W` are the mask and whitefield arrays respectively. 
    U and V are given by:
     
    .. math::
        
        U &= \text{max}(\text{ij}_\text{map}[0, i, j]) - \text{min}(\Delta ij_n[0]) + n_0 \\
        V &= \text{max}(\text{ij}_\text{map}[1, i, j]) - \text{min}(\Delta ij_n[1]) + m_0
    
    n0, m0 are given by:
    
    .. math::
        
        n_0 &= \text{max}_{i,j\in M}(\Delta ij[n, 0] - \text{ij}_\text{map}[0, i, j]) + 0.5  \\
        m_0 &= \text{max}_{i,j\in M}(\Delta ij[n, 1] - \text{ij}_\text{map}[1, i, j]) + 0.5  
    """
    # mask the pixel mapping
    ij     = np.array([pixel_map[0][mask], pixel_map[1][mask]])
    
    # choose the offset so that ij - dij_n + n0 > -0.5
    # for all masked pixels
    n0, m0 = -np.min(ij[0]) + np.max(dij_n[:, 0]), -np.min(ij[1]) + np.max(dij_n[:, 1])
    n0, m0 = n0-0.5, m0-0.5
    
    # determine the object-map domain
    shape   = (int(np.rint(np.max(ij[0]) - np.min(dij_n[:, 0]) + n0))+1, \
               int(np.rint(np.max(ij[1]) - np.min(dij_n[:, 1]) + m0))+1)
    I       = np.zeros(shape, dtype=np.float)
    overlap = np.zeros(shape, dtype=np.float)
    WW      = W**2
    
    for n in range(data.shape[0]):
        # define the coordinate mapping and round to int
        ss = np.rint((ij[0] - dij_n[n, 0] + n0)).astype(np.int)
        fs = np.rint((ij[1] - dij_n[n, 1] + m0)).astype(np.int)
        #
        I[      ss, fs] += (W*data[n])[mask]
        overlap[ss, fs] += WW[mask]
        
        #I[      i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += (mask*W*data[n])[ij[0], ij[1]]
        #overlap[i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += (mask*W**2     )[ij[0], ij[1]]
    
    overlap[overlap<1e-2] = -1
    m = (overlap > 0)
    
    I[m]  = I[m] / overlap[m]
    I[~m] = -1
    return I, n0, m0
