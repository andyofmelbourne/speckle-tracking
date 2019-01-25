import numpy as np

def make_pixel_map(z, z1, dz, roi, x_pixel_size, y_pixel_size, shape, dx=None, dy=None):
    r"""

    Parameters
    ----------
    z : float
        The distance between the focus and the detector in meters.

    z1 : float
        The average focus to sample distance
       
    dz : float
        (:math:`\delta z`) The difference between the average defocus and the defocus along the 
        fast and slow axes of the detector: 
        z_ss = defocus + dz, 
        z_fs = defocus - dz, 
        defocus > 0 --> sample downstream of focus

    roi : array_like
        (:math:`\text{roi}`) Length 4 list of integers e.g. roi = [10, 400, 23, 500], 
        indicates that most of the interesting data in a frame will 
        be in the region: frame[roi[0]:roi[1], roi[2]:roi[3]]
    
    shape : array_like
        Length 2 list of integers e.g. (100, 200) equal to the pixel
        dimensions of each detector frame.
    
    x_pixel_size : float
        (:math:`\Delta_{ss}`) The side length of a detector pixel in metres, along the slow
        scan axis.
    
    y_pixel_size : float
        (:math:`\Delta_{fs}`) The side length of a detector pixel in metres, along the fast
        scan axis.

    Returns
    -------
    pixel_map : ndarray
        Array of shape (2,) + shape containing the pixel mapping 
        between the object and a detector frame, 
        
        .. math:: 
        
            \text{ij}_\text{map} = \left(
                                     \frac{\Delta_{ss} (i-\text{roi}[0])}{\Delta x M_{ss}},
                                     \frac{\Delta_{fs} (j-\text{roi}[2])}{\Delta y M_{fs}} 
                                     \right)
        
        such that:
        
        .. math:: 
        
            I^\infty[\text{ij}_\text{map}[0, i, j], \text{ij}_\text{map}[1, i, j]] = I^{z_1}_{\phi}[i, j]
            
    pixel_map_inv : ndarray
        Array of shape (2,) + shape containing the pixel mapping 
        between a detector frame and the object, 
        such that I[pixel_map_inv[i,j]] = O[i,j].
        
        .. math:: 
        
            \text{ij}^{-1}_\text{map} = \left(
                                     \frac{\Delta x M_{ss}}{\Delta_{ss}}i + \text{roi}[0],
                                     \frac{\Delta y M_{fs}}{\Delta_{fs}}j + \text{roi}[2],
                                     \right)
        
        such that:
        
        .. math:: 
        
            I^\infty[i, j] = I^{z_1}_{\phi}[\text{ij}^{-1}_\text{map}[0, i, j], \text{ij}^{-1}_\text{map}[1, i, j]]

    dxy : tuple 
        Length 2 tuple containing the resolution along the slow and fast scan axes of the detector:
        
        .. math::
            
            \Delta x = \Delta y = \frac{\text{min}(\Delta_{ss}, \Delta_{fs})}{\text{min}(M_{ss}, M_{fs})}
    Notes
    -----
    We have:

    .. math:: 
        
        \mathbf{x}_\text{map} = \mathbf{x} \circ \left( \frac{z_1 + \delta z}{z + \delta z}, \frac{z_1 - \delta z}{z - \delta z} \right)
        = x \circ (M^{-1}_{ss}, M^{-1}_{fs})
    
    And if :

    .. math:: I^{z_1}_\phi(x, y) = I^\infty(\frac{x}{M_{ss}}, \frac{y}{M_{fs}}) 

    and
    
    .. math:: 
    
        I^\infty_{nm} &\equiv I^\infty(\Delta x(n - n_0), \Delta y(m - m_0))  \\
        I^{z_1}_{\phi, ij} &\equiv I^{z_1}_\phi(\Delta_{ss} (i - i_0), \Delta_{fs} (j - j_0)) 

    then:
        
    .. math:: 
    
        I^{z_1}_{\phi, ij} &= I^{z_1}_\phi(\Delta_{ss} (i - i_0), \Delta_{fs} (j - j_0)) 
                           = I^\infty\left(\frac{\Delta_{ss} (i - i_0)}{M_{ss}}, 
                                           \frac{\Delta_{fs} (j - j_0)}{M_{fs}}\right) \\
                          &= I^\infty\left(\Delta x \left(\frac{\Delta_{ss} (i - i_0)}{\Delta x M_{ss}} + n_0 - n_0\right),
                                           \Delta y \left(\frac{\Delta_{fs} (j - j_0)}{\Delta y M_{fs}} + m_0 - m_0\right) \right) \\
                          &= I^\infty_{\frac{\Delta_{ss} (i - i_0)}{\Delta x M_{ss}} + n_0, \frac{\Delta_{fs} (j - j_0)}{\Delta y M_{fs}} + m_0}

    So we have:
    
    .. math:: 
    
        \text{ij}_\text{map} = \left(
                                 \frac{\Delta_{ss} (i-\text{roi}[0])}{\Delta x M_{ss}},
                                 \frac{\Delta_{fs} (j-\text{roi}[2])}{\Delta y M_{fs}} 
                                 \right)
        
    Where:

    .. math::
        
        (i_0, j_0) &= \left((\text{roi}[1]-\text{roi}[0])/2 + \text{roi}[0], 
                            (\text{roi}[3]-\text{roi}[2])/2 + \text{roi}[2]\right) \\
        (n_0, m_0) &= \left(\frac{\Delta_{ss} (i_0-\text{roi}[0])}{\Delta x M_{ss}}, 
                            \frac{\Delta_{fs} (j_0-\text{roi}[2])}{\Delta y M_{fs}}\right)

    We have choosen :math:`i_0, j_0` so that :math:`x=0` corresponds to the centre of the 
    detector roi, and :math:`n_0, m_0` so that :math:`\text{ij}_\text{map}>0` for all
    pixels in the roi.
    """
    Mss, Mfs = (z + dz)/(z1 + dz), (z - dz)/(z1 - dz)
    
    if dx is None or dy is None :
        dx = dy = min(x_pixel_size / Mss, y_pixel_size / Mfs)
    
    i, j = np.arange(shape[0]), np.arange(shape[1])
    
    i_map  = x_pixel_size * (i) / (dx * Mss)
    j_map  = y_pixel_size * (j) / (dy * Mfs)
    ii, jj = np.meshgrid(i_map, j_map, indexing='ij')
    pixel_map = np.array([ii, jj])
    
    i, j = np.arange(int(i_map[roi[1]])), np.arange(int(j_map[roi[3]]))
    i_map  = (dx * Mss) * i / x_pixel_size  + roi[0]
    j_map  = (dy * Mfs) * j / y_pixel_size  + roi[2]
    ii, jj = np.meshgrid(i_map, j_map, indexing='ij')
    pixel_map_inv = np.array([ii, jj]) 
    
    return pixel_map, pixel_map_inv, (dx, dy)
