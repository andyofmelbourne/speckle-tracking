import tqdm
import numpy as np
import scipy.signal

def make_mask(data, thresh = 400, mask_gaps=False):
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
    
    """
    # testing maths in the doc-string
    #Notes
    #-----
    #.. math::
    #    
    #    y = x^2
    ms = np.zeros(data.shape[1:], dtype=np.float)
    for i in tqdm.trange(data.shape[0], desc='Making the mask'):
        ms += (data[i] - scipy.signal.medfilt(data[i], 3))**2
    
    var = scipy.signal.medfilt(np.var(data, axis=0))
    var[var==0] = 1
    
    mask = (ms/var) < thresh
    
    if mask_gaps :
        mask *= make_gap_mask(data)
    return mask

# mask pixels that are the same as their neighbours
def make_gap_mask(data):
    m = np.ones(data.shape[1:], dtype=np.bool)
    mss = np.sum(np.diff(data, axis=1), axis=0)==0
    i, j = np.where(mss)
    m[i, j]   = False
    m[i+1, j] = False

    mfs = np.sum(np.diff(data, axis=2), axis=0)==0
    i, j = np.where(mfs)
    m[i, j]   = False
    m[i, j+1] = False
    return m
