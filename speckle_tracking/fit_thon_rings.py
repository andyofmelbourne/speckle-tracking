import numpy as np
from scipy.ndimage import filters 
from scipy.stats import pearsonr

def fit_thon_rings(data, x_pixel_size, y_pixel_size, z, wav, mask, W, roi, centre=None, sig=10, edge_pix=5, window=30, rad_range=None, verbose=True):
    r"""Find the focus to sample distance by fitting Thon rings to power spectrum.

    This is done by generating a filtered power spectrum of the data. Then fitting
    concentric rings to this profile. The fitting parameters are then used to 
    estimate the horizontal and vertical focus to sample distance.

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

    centre : array_like, optional
        Length 2 list of integers designating the centre of the 
        Gaussian that is applied to the data before averaging. 
        Default value is None in which case the centre of roi 
        region is used.
    
    edge_pix : integer, optional
        Number of edge pixels to mask in the power spectrum. 
    
    window : int, optional
        The sidelength of a square used to flatten the power spectrum
        contrast through the use of min / max filters.
    
    rad_range : array_like or None, optional
        Length 2 list [min val., max val.] of the pixel radius within
        the power spectrum to fit the Thon rings. If None, then this
        is set to:
        [min(10, W.shape[0], W.shape[1]), max(W.shape[0], W.shape[1])//2]
    
    verbose : bool, optional
        print what I'm doing. 
    
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
        Contains diagnostic information:
        res['thon_display'] shows the thon rings and the fit rings in one 
        quadrant of the array.
    
    Notes
    -----
    This routine fits the following function to the power spectrum:
    
    .. math::

        f(q_{ss}, q_{fs}) = p(q) \left( \delta_\lambda \sin(\pi\lambda u^2) + 
                                    \beta_\lambda \cos(\pi\lambda u^2) \right)
    
    where :math:`p(q)` is a q dependent profile that depends on the details of the 
    object, :math:`\lambda` is the wavelength and u is given by:
    
    .. math::
        
        u(q_{ss}, q_{fs}) = z_{ss}^\text{eff} (M_{ss} q_{ss})^2 + 
                            z_y^\text{eff} (M_{fs} q_{fs})^2 
    
    .. math::
        z_{ss}^\text{eff} &= \left( \frac{1}{z_2} + \frac{1}{z_1+\delta z} \right)^{-1} &
        z_{fs}^\text{eff} &= \left( \frac{1}{z_2} + \frac{1}{z_1-\delta z} \right)^{-1} \\
        M_{ss} &= z_2  / z_{ss}^\text{eff} & 
        M_{fs} &= z_2  / z_{fs}^\text{eff}

    subject to:
    
    .. math:: z = z_1 + z_2
    """

    if verbose : print('fitting the defocus and astigmatism:\n')
    # generate the thon rings
    # offset centre to avoid panel edges
    thon = make_thon(data, mask, W, roi, sig=sig, centre=centre)
    
    # make an edge mask
    edge_mask = make_edge_mask(thon.shape, edge_pix)

    # flatten the thon rings with a min max filter
    thon_flat = flatten(thon, edge_mask, w=window, sig=0.)

    # fit the quadratic symmetry to account for astigmatism etc
    theta, scale_fs, res = fit_theta_scale(thon_flat, edge_mask)

    # fit the target function for thon rings
    if rad_range is None :
        rs = np.arange(min(10, W.shape[0], W.shape[1]), max(W.shape[0], W.shape[1])//2, 1)
    else :
        rs = np.arange(rad_range[0], rad_range[1], 1)
    
    c, bd, res2 = fit_sincos(res['im_rav'][rs], rs)
    
    # convert to physical units
    ###########################
    
    # solve for z1 and dz
    a = (thon.shape[0] * x_pixel_size)**2 * c / (np.pi * wav)
    b = (thon.shape[1] * y_pixel_size)**2 * c / (np.pi * wav) * scale_fs

    z1 = (-a*b + 2*z**2 + np.sqrt(a**2*b**2 + a**2*z**2 - 2*a*b*z**2 + b**2*z**2))/(a + b + 2*z)
    dz = (a*b - np.sqrt(a**2*b**2 + a**2*z**2 - 2*a*b*z**2 + b**2*z**2))/(a - b)

    # calculate thon rings
    thon_calc = calculate_thon(z, z1, dz, x_pixel_size, y_pixel_size, thon.shape, wav, bd)

    # overlay with forward calculation for display
    thon = make_thon(data, mask, W, roi, sig=None, centre=centre)
    thon_dis = np.log(thon)**0.2
    thon_dis = (thon_dis-thon_dis.min())/(thon_dis-thon_dis.min()).max()
    thon_dis[:thon.shape[0]//2, :thon.shape[1]//2] = thon_calc[:thon.shape[0]//2, :thon.shape[1]//2]
    thon_dis = np.fft.fftshift(thon_dis)
    return z1, dz, {'thon_display': thon_dis}


def make_thon(data, mask, W, roi=None, sig=None, centre=None):
    if sig is None :
        sig = data.shape[-1]//4
    
    if roi is not None and centre is None :
        centre = [(roi[1]-roi[0])/2 + roi[0], 
                  (roi[3]-roi[2])/2 + roi[2]]

    reg = mk_2dgaus(data.shape[1:], sig, centre)
    
    thon = np.zeros(data.shape[1:], dtype=np.float)
    for i in range(data.shape[0]):
        # mask data and fill masked pixels
        temp = mask * data[i] / W
        temp[mask==False] = 1.
        temp *= reg
        thon += np.abs(np.fft.fftn(np.fft.fftshift(temp)))**2  
    
    return thon

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

def make_edge_mask(shape, edge, is_fft_shifted=True):
    mask = np.ones(shape, dtype=np.bool)
    mask[:edge,  :] = False
    mask[-edge:, :] = False
    mask[:,  :edge] = False
    mask[:, -edge:] = False
    if not is_fft_shifted :
        mask = np.fft.fftshift(mask)
    return mask

def flatten(im, mask, w=5, sig=2.):
    """
    out = (im - min) * min / max
    """
    if sig > 0. :
        out = filters.gaussian_filter(im * mask, sig, mode = 'wrap')
        
        # normalise
        m   = filters.gaussian_filter(mask.astype(np.float), sig, mode = 'wrap')
        m[m==0] = 1
        out /= m
    else :
        out = mask * im
    
    imin = filters.minimum_filter(out, size = w, mode = 'wrap')
    imax = filters.maximum_filter(out, size = w, mode = 'wrap')
    c   = (imax - imin)
    out = (out - imin) 

    c[c==0] = 1
    out /= c
    return out

def fit_theta_scale(im, mask):
    ts = np.linspace(0, np.pi/2, 50, endpoint=False)
    ds = np.linspace(0.5, 1.5, 50)

    shape = im.shape
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')
    ij = np.vstack((np.ravel(i), np.ravel(j)))
    
    def forward(t, d):
        R = np.array([[np.cos(t), np.sin(t)], [-np.sin(t), np.cos(t)]])
        D = np.diag([1, d])
        
        r = np.sqrt(np.sum(np.dot(D, np.dot(R, ij)**2), axis=0).reshape(shape))
        im_rav = radial_symetry(im, r, mask=mask)
        im2    = im_rav[r.astype(np.int)].reshape(shape)
        return im2, r, im_rav
    
    def fun(t,d):
        return np.sum( mask * (forward(t, d)[0] - im)**2 )
    
    error = np.empty(ts.shape + ds.shape, dtype=np.float)
    error.fill(np.inf)
    for ti, t in enumerate(ts) :
        for di, d in enumerate(ds) :
            error[ti, di] = fun(t, d)
    
    ti, di  = np.unravel_index(np.argmin(error), error.shape)
    t, d    = ts[ti], ds[di]
    im_sym, r_vals, im_rav = forward(t, d)
    res = {'error_map': error, 'im_sym': im_sym, 'r_vals': r_vals, 'im_rav': im_rav}
    return t, d, res

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

def fit_sincos(f, r):
    vis = f.max() - f.min()
    def forward(a, b):
        return (np.sin(a*r**2) + b * np.cos(a*r**2))**2 * vis / max(1., b**2)
    
    def fun(a, b):
        err, _ = pearsonr(f,forward(a,b))
        return err
    
    # set the maximum a so that we don't alias
    dr = np.abs(r[-1] - r[-2])
    rr = np.abs(r[-1] - r[0])
    amax = np.pi / (2. * dr * np.max(r))
    amin = 2. * np.pi / (rr * np.max(r))

    a_s = np.linspace(amin, amax, 1000, endpoint=False)
    b_s = np.linspace(-1, 1, 100)
    
    error = np.empty(a_s.shape + b_s.shape, dtype=np.float)
    error.fill(np.inf)
    for ai, a in enumerate(a_s) :
        for bi, b in enumerate(b_s) :
            error[ai, bi] = fun(a, b)
            #print(ai, bi, error[ai, bi])
    
    ai, bi  = np.unravel_index(np.argmax(error), error.shape)
    a, b    = a_s[ai], b_s[bi]
    res = {'error_map': error, 'fit': forward(a, b)}
    return a, b, res

def calculate_thon(z, z1, dz, x_pixel_size, y_pixel_size, shape, wav, bd):
    z2 = z-z1
    z_ss = 1/(1/z2 + 1/(z1+dz))
    z_fs = 1/(1/z2 + 1/(z1-dz))
    M_ss = z2 / z_ss
    M_fs = z2 / z_fs
    i = M_ss * np.fft.fftfreq(shape[0], x_pixel_size)
    j = M_fs * np.fft.fftfreq(shape[1], y_pixel_size)
    i, j = np.meshgrid(i, j, indexing='ij')
    
    q2 = np.pi * wav * (z_ss * i**2 + z_fs * j**2)
    thon = (np.sin(q2) + bd * np.cos(q2))**2
    return thon
