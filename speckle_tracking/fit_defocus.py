import numpy as np

from .fit_thon_rings import fit_thon_rings


def fit_defocus(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi, **kwargs):
    """Estimate the focus to sample distance.
    
    This routine uses speckle_tracking.fit_thon_rings to estimate the defocus.
    
    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L). Where
        N = number of frames
        M = number of pixels along the slow scan axis of the detector
        L = number of pixels along the fast scan axis of the detector
    
    x_pixel_size : float
        The side length of a detector pixel in metres, along the slow
        scan axis.

    y_pixel_size : float
        The side length of a detector pixel in metres, along the fast
        scan axis.

    z : float
        The distance between the focus and the detector in meters.

    wav : float
        The wavelength of the imaging radiation in metres.

    W : ndarray
        The whitefield array of shape (M, L). This is the image one 
        obtains without a sample in place.

    roi : array_like
        Length 4 list of integers e.g. roi = [10, 400, 23, 500], 
        indicates that most of the interesting data in a frame will 
        be in the region: frame[roi[0]:roi[1], roi[2]:roi[3]]

    kwargs : dict 
        keyword arguments that are passed on to any functions called.
    
    Returns
    -------
    defocus : float
        The average focus to sample distance

    dz : float
        The difference between the average defocus and the defocus along the 
        fast and slow axes of the detector: 
        z_ss = defocus + dz
        z_fs = defocus - dz
        defocus > 0 --> sample downstream of focus

    res : dict
        Contains diagnostic information
    
    See Also
    --------
    fit_thon_rings
    """
    return fit_thon_rings(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi, **kwargs)


