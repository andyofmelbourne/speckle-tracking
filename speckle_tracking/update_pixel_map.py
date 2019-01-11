import numpy as np

def update_pixel_map():
    r"""
    Notes
    -----
    .. math:: 
    
        \varepsilon[i, j] = \bigg(\sum_n I^\infty[i- \Delta ij[n, 0] + n_0, j - \Delta ij[n, 1] + m_0] - \\
        I^{z_1}_{\phi}[n, \text{ij}^{-1}_\text{map}[0, i, j], \text{ij}^{-1}_\text{map}[1, i, j]]\bigg)^2

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
    
    for ii in range(data.shape[1]):
        for jj in range(data.shape[2]):
            u, v = ij[0, ii, jj], ij[1, ii, jj]
            I[      i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += mask[u, v]*W[u, v]*data[:, u, v]
            overlap[i + nm0[0]-dij_nr[n, 0], j + nm0[1]-dij_nr[n, 1]] += mask[u, v]*W[u, v]**2  
    
    overlap[overlap<1e-2] = -1
    m = (overlap > 0)
    
    I[m]  = I[m] / overlap[m]
    I[~m] = -1
    return I
