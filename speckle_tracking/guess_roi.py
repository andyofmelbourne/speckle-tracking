import numpy as np

def guess_roi(W, verbose=True):
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
    roi = [0, 0, 0, 0]
    
    # left ss intercept
    x = np.arange(W.shape[0])
    y = np.cumsum(np.sum(W, axis=1))
    p = np.polyfit(x, y, 2)
    roi[0] = int(round(-p[2]/p[1]))
    # y = mx + c, 0 = mx + c, x = -c / m

    # right ss intercept
    y = np.cumsum(np.sum(W, axis=1)[::-1])
    p = np.polyfit(x, y, 2)
    roi[1] = W.shape[0] - int(round(-p[2]/p[1])) 
    
    # left fs intercept
    x = np.arange(W.shape[1])
    y = np.cumsum(np.sum(W, axis=0))
    p = np.polyfit(x, y, 2)
    roi[2] = int(round(-p[2]/p[1]))
    # y = mx + c, 0 = mx + c, x = -c / m

    # right ss intercept
    y = np.cumsum(np.sum(W, axis=0)[::-1])
    p = np.polyfit(x, y, 2)
    roi[3] = W.shape[1] - int(round(-p[2]/p[1])) 
    
    if verbose : print('guessing the region of interest:', roi)
    return roi

