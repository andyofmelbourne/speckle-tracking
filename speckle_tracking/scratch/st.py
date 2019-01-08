import speckle_tracking as st
import h5py
import numpy as np
import pyqtgraph as pg
import scipy.signal
from scipy.ndimage import filters 


def make_thon(data, mask, W, roi=None, sig=None, centre=None):
    if sig is None :
        sig = data.shape[-1]//4
    
    if roi is not None and centre is None :
        centre = [(roi[1]-roi[0])/2 + roi[0], 
                  (roi[3]-roi[2])/2 + roi[2]]

    reg = st.mk_2dgaus(data.shape[1:], sig, centre)
    
    thon = np.zeros(data.shape[1:], dtype=np.float)
    for i in range(data.shape[0]):
        # mask data and fill masked pixels
        temp = mask * data[i] / W
        temp[mask==False] = 1.
        temp *= reg
        thon += np.abs(np.fft.fftn(np.fft.fftshift(temp)))**2  
    
    return np.fft.fftshift(thon)

def make_whitefield(data, mask):
    whitefield = np.median(data, axis=0)
    
    mask2  = mask.copy()
    mask2 *= (whitefield != 0) 

    # fill masked pixels whith neighbouring values
    whitefield[~mask2] = scipy.signal.medfilt(mask2*whitefield, 5)[~mask2]
    whitefield[whitefield==0] = 1.
    return whitefield

def find_symmetry(thon):
    """
    r^2 = ax^2 + dy^2 + 2bxy
    """
    shape = thon.shape
    i = np.fft.fftfreq(shape[0]) * shape[0]
    j = np.fft.fftfreq(shape[1]) * shape[1]
    i, j = np.meshgrid(i, j, indexing='ij')

    mask = np.ones(thon.shape, dtype=np.bool)
    edge = 10
    mask[:edge,  :] = False
    mask[-edge:, :] = False
    mask[:,  :edge] = False
    mask[:, -edge:] = False
    
    ds = np.linspace(0.8, 2.0, 50)
    bs = np.linspace(-0.05, 0.05, 20)

    var = np.zeros(ds.shape + bs.shape, dtype=np.float)
    for ii, d in enumerate(ds) : 
        print(ii, d)
        for jj, b in enumerate(bs) : 
            rs   = np.sqrt(i**2 + d*j**2 + 2*b*i*j) 
            rav  = st.radial_symetry(thon, rs, mask)
            
            var[ii, jj] = np.var(rav[20:200])
    
    ii, jj = np.unravel_index(np.argmax(var), var.shape)

    d = ds[ii]
    b = bs[jj]
    rs   = np.sqrt(i**2 + d*j**2 + 2*b*i*j) 
    rav  = st.radial_symetry(thon, rs, mask)
    print(d, b)
    return var, rav

def get_r_theta(shape, d, is_fft_shifted = True):
    i = np.fft.fftfreq(shape[0], d[0]) 
    j = np.fft.fftfreq(shape[1], d[1]) 
    i, j = np.meshgrid(i, j, indexing='ij')
    qs   = np.sqrt(i**2 + j**2)
    
    ts = np.arctan2(i, j)
    if is_fft_shifted is False :
        qs = np.fft.fftshift(qs)
        ts = np.fft.fftshift(ts)

    return qs, ts

def fit_thon(q, z1, zD, wav, thon_rav, pr=[40,160]):
    import scipy.stats
    # range from the second minima to the 
    # horizontal egdge of the detector
    #istart, istop = zi[1], int(len(thon_rav)/np.sqrt(2))
    rad_range = pr
    def forward(z):
        return np.sin( np.pi * wav * (zD - z1) * zD/z1 * q**2 )**2
        
    def fun(z, f):
        err, _ = scipy.stats.pearsonr(f[rad_range[0]:rad_range[1]], 
                                forward(z)[rad_range[0]:rad_range[1]])
        return err

    z1s  = np.linspace(z1*0.5, z1*1.5, 1000)
    errs = np.zeros_like(z1s)
    for i, z in enumerate(z1s) :
        # make the target function
        bac, env = fit_envolopes_min_max(q, z, zD, wav, thon_rav)
        f = (thon_rav-bac)/env 
            
        err = fun(z1, f)
        errs[i] = err
        #print(i, err)

    i = np.argmax(errs)
    z1out = z1s[i]
    err   = errs[i]
    print('best defocus, err:', z1out, err)
    bac, env = fit_envolopes_min_max(q, z1out, zD, wav, thon_rav)
    return forward(z1out), bac, env, errs, z1s

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

def fit_sincos(f, r):
    vis = f.max() - f.min()
    def forward(a, b):
        return (np.sin(a*r**2) + b * np.cos(a*r**2))**2 * vis / max(1., b**2)
    
    def fun(a, b):
        err, _ = scipy.stats.pearsonr(f,forward(a,b))
        return err
    
    # set the maximum a so that we don't alias
    dr = np.abs(r[-1] - r[-2])
    rr = np.abs(r[-1] - r[0])
    amax = np.pi / (2. * dr * np.max(r))
    amin = 2. * np.pi / (rr * np.max(r))

    print(amin, amax)
    
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

def fit_envolopes_min_max(q, z1, zD, wav, thon_rav):
    """
    estimate background and envelope by looking for the 
    min / max value within each period
    this ensures that the env > 0
    """
    t  = z1 / (wav * zD * (zD-z1))

    def get_ql(l):
        return np.sqrt( t * (l + 0.25))

    #nmax = int(q[-1]**2  / t)
    #mmax = int(q[-1]**2  / t - 0.5)
    lmax = int(q[-1]**2  / t - 0.25)
    
    #qn = np.sqrt( t *  np.arange(nmax+1))
    #qm = np.sqrt( t * (np.arange(mmax+1) + 0.5))
    qls     = np.sqrt( t * (np.arange(lmax+1) + 0.25))
    qls_mid = np.sqrt( t * (np.arange(lmax+1) + 0.75))
    qls_i   = np.searchsorted(q, qls)
    
    # the min value between l and l+1
    thon_mins = [np.min(thon_rav[qls_i[i]:qls_i[i+1]]) for i in range(len(qls)-1)]
    thon_maxs = [np.max(thon_rav[qls_i[i]:qls_i[i+1]]) for i in range(len(qls)-1)]
    thon_mins = np.array(thon_mins)
    thon_maxs = np.array(thon_maxs)

    from scipy.interpolate import interp1d
    fmins = interp1d(qls_mid[:-1], thon_mins, kind='cubic', fill_value='extrapolate', bounds_error=False)
    fmaxs = interp1d(qls_mid[:-1], thon_maxs, kind='cubic', fill_value='extrapolate', bounds_error=False)

    b   = fmins(q)
    env = fmaxs(q) - b

    return b, env

def fit_envolopes_min_max_linear(q, z1, zD, wav, thon_rav):
    """
    estimate background and envelope by looking for the 
    min / max value within each period
    this ensures that the env > 0
    """
    t  = z1 / (wav * zD * (zD-z1))

    def get_ql(l):
        return np.sqrt( t * (l + 0.25))

    #nmax = int(q[-1]**2  / t)
    #mmax = int(q[-1]**2  / t - 0.5)
    lmax = int(q[-1]**2  / t - 0.25)
    
    #qn = np.sqrt( t *  np.arange(nmax+1))
    #qm = np.sqrt( t * (np.arange(mmax+1) + 0.5))
    qls     = np.sqrt( t * (np.arange(lmax+1) + 0.25))
    qls_mid = np.sqrt( t * (np.arange(lmax+1) + 0.75))
    qls_i   = np.searchsorted(q, qls)
    
    # the min value between l and l+1
    thon_mins = [np.min(thon_rav[qls_i[i]:qls_i[i+1]]) for i in range(len(qls)-1)]
    thon_maxs = [np.max(thon_rav[qls_i[i]:qls_i[i+1]]) for i in range(len(qls)-1)]
    thon_mins = np.array(thon_mins)
    thon_maxs = np.array(thon_maxs)

    b   = interp(q, qls_mid[:-1], thon_mins)
    env = interp(q, qls_mid[:-1], thon_maxs-thon_mins) 
    return b, env


def fit_envolopes(q, z1, zD, wav, thon_rav):
    t = z1 / (wav * zD * (zD-z1))
    nmax = int(q[-1]**2  / t)
    mmax = int(q[-1]**2  / t - 0.5)
    qz = np.sqrt( t *  np.arange(nmax+1))
    qo = np.sqrt( t * (np.arange(mmax+1) + 0.5))

    zi = np.searchsorted(q, qz)
    oi = np.searchsorted(q, qo)
    
    # get the value of thon_rav at these points
    thonz = interp(qz, q, thon_rav)
    thono = interp(qo, q, thon_rav)
    thonz[0] = 2*thono[0]
    
    # now estimate the background from the zeros
    b  = interp(q, qz, thonz)
    
    # now estimate the envelope
    en = interp(q, qo, thono) - b
    return b, en, zi, oi


def interp(xn, xo, y):
    if len(y) != len(xo) :
        raise ValueError('xo and y must be the same length')
    
    # trilinear interp
    if type(xn) is float :
        # find the largest xo smaller than xn
        i = np.searchsorted(xo, xn)
        if i == 0 :
            out = y[0]
        elif i == len(y) :
            out = y[-1]
        else :
            out = (xo[i]-xn)*y[i-1] + (xn-xo[i-1])*y[i]
            out /= xo[i]-xo[i-1]
    else :
        i  = np.searchsorted(xo, xn)
        im = i-1  
        
        # keep in bounds
        m = np.ones(i.shape, dtype=np.bool)
        m[i<=0]      = False
        m[i>=len(y)] = False

        out    = np.zeros(xn.shape, dtype = np.float)
        
        # lever interpolation
        out[m] = (xo[i[m]]-xn[m])*y[im[m]] + (xn[m]-xo[im[m]])*y[i[m]]
        
        # divide by step size 
        d              = xo[i[m]]-xo[im[m]]
        out[m]         = out[m]/d
        out[i<=0]      = y[0]
        out[i>=len(y)] = y[-1]
    return out

def make_edge_mask(shape, edge, is_fft_shifted=True):
    mask = np.ones(shape, dtype=np.bool)
    mask[:edge,  :] = False
    mask[-edge:, :] = False
    mask[:,  :edge] = False
    mask[:, -edge:] = False
    if not is_fft_shifted :
        mask = np.fft.fftshift(mask)
    return mask

def make_thon_rav(thon, pix_size, mask, is_fft_shifted=True):
    # get the q and theta grid
    q, t = get_r_theta(thon.shape, pix_size, is_fft_shifted)
    
    # scale q so that we sample the pixels well
    r     = q * np.sqrt(thon.shape[0]**2 + thon.shape[1]**2)/2 / q.max()
    
    # now get the radial average
    thon_rav = st.radial_symetry(thon, r, mask)
    
    # now scale the 1d q-scale
    q_rav    = np.linspace(0, q.max(), thon_rav.shape[0])
    return q_rav, thon_rav

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
    import speckle_tracking as st 
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
        im_rav = st.radial_symetry(im, r, mask=mask)
        im2    = im_rav[r.astype(np.int)].reshape(shape)
        return im2, r, im_rav
    
    def fun(t,d):
        return np.sum( mask * (forward(t, d)[0] - im)**2 )
    
    error = np.empty(ts.shape + ds.shape, dtype=np.float)
    error.fill(np.inf)
    for ti, t in enumerate(ts) :
        for di, d in enumerate(ds) :
            error[ti, di] = fun(t, d)
            print(ti, di, error[ti, di])

    ti, di  = np.unravel_index(np.argmin(error), error.shape)
    t, d    = ts[ti], ds[di]
    im_sym, r_vals, im_rav = forward(t, d)
    res = {'error_map': error, 'im_sym': im_sym, 'r_vals': r_vals, 'im_rav': im_rav}
    return t, d, res

# extract data
f = h5py.File('siemens_star.cxi', 'r')

data  = f['/entry_1/data_1/data'][()]
basis = f['/entry_1/instrument_1/detector_1/basis_vectors'][()]
z     = f['/entry_1/instrument_1/detector_1/distance'][()]
x_pixel_size = f['/entry_1/instrument_1/detector_1/x_pixel_size'][()]
y_pixel_size = f['/entry_1/instrument_1/detector_1/y_pixel_size'][()]
wav          = f['/entry_1/instrument_1/source_1/wavelength'][()]
translations = f['/entry_1/sample_3/geometry/translation'][()]

mask  = f['results/mask'][()]
f.close()

#mask  = st.make_mask(data)

# estimate the whitefield
W = st.make_whitefield(data, mask)

# estimate the region of interest
roi = st.guess_roi(W)

# generate the thon rings
# offset centre to avoid panel edges
centre = [(roi[1]-roi[0])/2 + roi[0] + 20, 
          (roi[3]-roi[2])/2 + roi[2] + 0]
thon = np.fft.ifftshift(make_thon(data, mask, W, roi, sig=10, centre=centre))

# make an edge mask
edge_mask = make_edge_mask(thon.shape, 5)

# flatten the thon rings with a min max filter
thon_flat = flatten(thon, edge_mask, w=30, sig=0.)

# fit the quadratic symmetry to account for astigmatism etc
theta, scale_fs, res = fit_theta_scale(thon_flat, edge_mask)

# fit the target function for thon rings
rs = np.arange(20, 200, 1)
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

thon = np.fft.ifftshift(make_thon(data, mask, W, roi, sig=None, centre=centre))
thon_dis = np.log(thon)**0.2
thon_dis = (thon_dis-thon_dis.min())/(thon_dis-thon_dis.min()).max()
thon_dis[:thon.shape[0]//2, :thon.shape[1]//2] = thon_calc[:thon.shape[0]//2, :thon.shape[1]//2]
thon_dis = np.fft.fftshift(thon_dis)
