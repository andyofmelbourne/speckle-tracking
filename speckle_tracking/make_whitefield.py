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
    whitefield = fill_bad(whitefield, mask2, 4.)
    return whitefield

def fill_bad(pm, mask, sig): 
    out = np.zeros_like(pm)
    
    from scipy.ndimage.filters import gaussian_filter
    out   = gaussian_filter(mask * pm, sig, mode = 'constant', truncate=20.)
    norm  = gaussian_filter(mask.astype(np.float), sig, mode = 'constant', truncate=20.)
    norm[norm==0.] = 1.
    out2 = pm.copy()
    out2[~mask] = out[~mask] / norm[~mask]
    return out2
