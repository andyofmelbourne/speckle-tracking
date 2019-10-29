
def docstring_glossary():
    r"""
    
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
    
    x_pixel_size : float
        The side length of a detector pixel in metres :math:`\Delta_{ss}`, 
        along the slow scan axis.
    
    y_pixel_size : float
        The side length of a detector pixel in metres :math:`\Delta_{fs}`, 
        along the fast scan axis.
    
    z : float
        The distance between the focus and the detector in metres
        :math:`z = z_1 + z_2`.
    
    wav : float
        The wavelength of the imaging radiation in metres :math:`\lambda`.
    
    W : ndarray, float, (M, L)
        The whitefield image :math:`W`. This is the image one obtains without a 
        sample in place.
    
    roi : array_like, int, (4,)
        Length 4 list of integers e.g. roi = [10, 400, 23, 500], 
        indicates that most of the interesting data in a frame will 
        be in the region: frame[roi[0]:roi[1], roi[2]:roi[3]]
    
    defocus : float
        The average focus to sample distance :math:`z_1`.
    
    dz : float
        The difference between the average defocus and the defocus along the 
        fast and slow axes of the detector: 
        
        .. math::
            
            z_{ss} &= z_1 + dz \\
            z_{fs} &= z_1 - dz
        
        :math:`z_1 > 0` indicates that the sample is downstream of focus.
    
    O : ndarray, float, (U, V)
        This is essentially a defocused image of the object :math:`O` 
        or :math:`I^\infty`. It is the image one would obtain with plane 
        wave illumination with the detector placed some distance from the 
        sample. 
        
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
    
    pixel_map : ndarray, float, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object :math:`ij_\text{map}`, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]

    dij_n : ndarray, float, (N, 2)
        An array containing the sample shifts for each detector image in pixel units
        :math:`\Delta ij_n`.
    """
    pass
