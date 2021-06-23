import numpy as np

def guess_roi(W, verbose=False):
    """Find the rectangular region that contains most of the whitefield.

    Parameters
    ----------
    W : ndarray
        The whitefield, that is the image one obtains without a 
        sample in place.
    
    verbose : bool, optional
        print what I'm doing.
    
    Returns
    -------
    roi : list
        e.g. roi = [10, 400, 23, 500], indicates that most of the 
        interesting data in a frame will be in the region:
        frame[roi[0]:roi[1], roi[2]:roi[3]]
    """
    cumsum = np.cumsum(np.sum(W, axis=1))
    i      = np.arange(cumsum.shape[0])
    i, j   = np.meshgrid(i, i, indexing='ij')
    err    = 4*(1 - (cumsum[j]-cumsum[i])/cumsum[-1]) + (j-i)/cumsum.shape[0]
    left, right = np.unravel_index(np.argmin(err), i.shape)

    cumsum = np.cumsum(np.sum(W, axis=0))
    i      = np.arange(cumsum.shape[0])
    i, j   = np.meshgrid(i, i, indexing='ij')
    err    = 4*(1 - (cumsum[j]-cumsum[i])/cumsum[-1]) + (j-i)/cumsum.shape[0]
    bottom, top = np.unravel_index(np.argmin(err), i.shape)
    
    roi = [left, right, bottom, top]
    if verbose : print('guessing the region of interest:', roi)
    return roi

