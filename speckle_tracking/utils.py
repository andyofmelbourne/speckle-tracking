import numpy as np

def mk_2dgaus(shape, sig, centre = None):
    if centre is None :
        centre = [shape[0]//2, shape[1]//2]
    if sig is not None : 
        x = np.arange(shape[0]) - centre[0]
        x = np.exp( -x**2 / (2. * sig**2))
        y = np.arange(shape[1]) - centre[1]
        y = np.exp( -y**2 / (2. * sig**2))
        reg = np.outer(x, y)
    else :
        reg = 1
    return reg

def get_r_theta(shape, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    rs   = np.sqrt(i**2 + j**2)
    
    ts = np.arctan2(i, j)
    if is_fft_shifted is False :
        rs = np.fft.fftshift(rs)
        ts = np.fft.fftshift(ts)
    return rs, ts

def radial_symetry(background, rs, mask=None):
    ########### Find the radial average
    # mask zeros from average
    if mask is None : 
        mask = np.ones(background.shape, dtype=np.bool)

    rs = rs.astype(np.int16).ravel()
    
    # get the r histogram
    r_hist = np.bincount(rs, mask.ravel().astype(np.int16))
    # get the radial total 
    r_av = np.bincount(rs, background.ravel() * mask.ravel())
    # prevent divide by zero
    nonzero = np.where(r_hist != 0)
    zero    = np.where(r_hist == 0)
    # get the average
    r_av[nonzero] = r_av[nonzero] / r_hist[nonzero].astype(r_av.dtype)
    r_av[zero]    = 0
    
    ########### Make a large background filled with the radial average
    #background = r_av[rs].reshape(background.shape)
    return r_av
