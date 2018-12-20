import numpy as np
import scipy.signal

def make_mask(data, thresh = 400):
    """Make a binary True/False mask from input data
    
    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L).
    
    thresh : float, optional
        The threshold value of the statistical measure below which 
        pixels are considered to be good (mask == False).
    
    Returns
    -------
    mask : ndarray
        Boolean array of shape (M, L), where True indicates a good
        pixel and False a bad pixel.
    
    Notes
    -----
    .. math::
        
        y = x^2
    """
    ms = np.zeros(data.shape[1:], dtype=np.float)
    for i in range(data.shape[0]):
        print(i, '/', data.shape[0])
        ms += (data[i] - scipy.signal.medfilt(data[i], 3))**2
    
    var = scipy.signal.medfilt(np.var(data, axis=0))
    var[var==0] = 1
    
    mask = (ms/var) < thresh
    return mask
