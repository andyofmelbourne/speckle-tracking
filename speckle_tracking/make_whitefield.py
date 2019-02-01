import numpy as np
import scipy.signal

def make_whitefield(data, mask, verbose=True):
    """Estimate the image one would obtain without the sample in the beam.

    This is done by taking the median value at each pixel along the first 
    axis, then we try to fill in bad / zero pixels (where mask == False).

    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L).

    mask : ndarray
        Boolean array of shape (M, L), where True indicates a good
        pixel and False a bad pixel.
    
    verbose : bool, optional
        print what I'm doing.
    
    Returns
    -------
    W : ndarray
        Float array of shape (M, L) containing the estimated whitefield.
    """
    if verbose: print('Making the whitefield')
    
    whitefield = np.median(data, axis=0)
    
    mask2  = mask.copy()
    mask2 *= (whitefield != 0) 

    # fill masked pixels whith neighbouring values
    whitefield[~mask2] = scipy.signal.medfilt(mask2*whitefield, 5)[~mask2]
    whitefield[whitefield==0] = 1.
    return whitefield
