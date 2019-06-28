import numpy as np

from .fit_thon_rings import fit_thon_rings
from .make_object_map import make_object_map
from .make_pixel_map import make_pixel_map
from .calc_error import calc_error
from .make_pixel_translations import make_pixel_translations


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

    verbose : bool, optional
        print what I'm doing. 
    
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
    fit_defocus_registration
    """
    defocus, res = fit_thon_rings(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi, **kwargs)
    
    if 'basis' in kwargs and 'translations' in kwargs :
        defocus1, res1 = fit_defocus_registration(
                               data, x_pixel_size, y_pixel_size, z, 
                               wav, mask, W, roi, kwargs['basis'], 
                               kwargs['translations'], window=window)
        
        # choose the one that gives the least error
        dij_n0 = make_pixel_translations(
                kwargs['translations'], kwargs['basis'], res['defocus_ss'], res['defocus_fs'])
        dz0 = res['astigmatism']
        u0, uinv, dxy = make_pixel_map(z, defocus, dz0, roi, x_pixel_size, y_pixel_size, mask.shape)
        O0, n00, m00  = make_object_map(data, mask, W, dij_n0, u0)
        error0        = calc_error(data, mask, W, dij_n0, O0, u0, n00, m00)[0]
        print('Thon ring error:', error0)
        
        dij_n1 = make_pixel_translations(
                kwargs['translations'], kwargs['basis'], res1['defocus_ss'], res1['defocus_fs'])
        dz1 = res1['astigmatism']
        u1, uinv, dxy = make_pixel_map(z, defocus1, dz1, roi, x_pixel_size, y_pixel_size, mask.shape)
        O1, n01, m01  = make_object_map(data, mask, W, dij_n1, u1)
        error1        = calc_error(data, mask, W, dij_n1, O1, u1, n01, m01)[0]
        print('Fit by registration error:', error1)

        if error1 > error0 :
            print('lower error for fit by registration')
            defocus = defocus1
            res     = res1
    return defocus, res


