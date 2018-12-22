import speckle_tracking as st
import h5py
import numpy as np
import pyqtgraph as pg
import scipy.signal


def make_thon(data, mask, W, roi=None):
    sig = data.shape[-1]//4
    if roi is not None :
        centre = [(roi[1]-roi[0])/2 + roi[0], 
                  (roi[3]-roi[2])/2 + roi[2]]
    else :
        centre = None
    
    reg = st.mk_2dgaus(data.shape[1:], sig, centre)
    
    thon = np.zeros(data.shape[1:], dtype=np.float)
    for i in range(data.shape[0]):
        thon += np.abs(np.fft.fftn(np.fft.fftshift(mask * reg * data[i] / W)))**2
    
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

def fit_thon(q, z1, zD, wav, thon_rav):
    import scipy.stats
    def fun(z):
        err, _ = scipy.stats.pearsonr(thon_1d[rad_range[0]:rad_range[1]], 
                                forward(z[0])[rad_range[0]:rad_range[1]])

    z1s  = np.linspace(z1*0.5, z1*1.5, 300)
    errs = np.zeros_like(z1s)
    for i, z in enumerate(z1s) :
        bac, env, zi, oi = fit_envolopes(q, z, zD, wav, thon_rav)
        
        f = (thon_rav-bac)/env 
        
        # range from the second minima to the 
        # horizontal egdge of the detector
        istart, istop = zi[1], int(len(thon_rav)/np.sqrt(2))
            
        s = np.sin( np.pi * wav * (zD - z1) * zD/z1 * q**2 )**2
        err, _ = scipy.stats.pearsonr( s[istart:istop], f[istart:istop] )
        errs[i] = err
        print(i, err)

    i = np.argmax(errs)
    z1out = z1s[i]
    err   = errs[i]
    print('best defocus, err:', z1out, err)
    bac, env, zi, oi = fit_envolopes(q, z1out, zD, wav, thon_rav)
    return np.sin( np.pi * wav * (zD - z1out) * zD/z1out * q**2 )**2, bac, env, errs, z1s

def fit_envolopes_min_max(q, z1, zD, wav, thon_rav):
    """
    estimate background and envelope by looking for the 
    min / max value within each period
    this ensures that the env > 0
    """
    t  = z1 / (wav * zD * (zD-z1))
    nmax = int(q[-1]**2  / t)
    mmax = int(q[-1]**2  / t - 0.5)
    lmax = int(q[-1]**2  / t - 0.25)
    
    qn = np.sqrt( t *  np.arange(nmax+1))
    qm = np.sqrt( t * (np.arange(mmax+1) + 0.5))
    ql = np.sqrt( t * (np.arange(lmax+1) + 0.25))


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
    q, t = get_r_theta(thon.shape, [x_pixel_size, y_pixel_size], False)
    
    # scale q so that we sample the pixels well
    r     = q * np.sqrt(thon.shape[0]**2 + thon.shape[1]**2)/2 / q.max()
    
    # now get the radial average
    thon_rav = st.radial_symetry(thon, r, mask)
    
    # now scale the 1d q-scale
    q_rav    = np.linspace(0, q.max(), thon_rav.shape[0])
    return q_rav, thon_rav

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


W = st.make_whitefield(data, mask)

roi = st.guess_roi(W)

thon = np.fft.fftshift(make_thon(data, mask, W, roi))

edge_mask = make_edge_mask(thon.shape, 5)

q_rav, thon_rav = make_thon_rav(thon, [x_pixel_size, y_pixel_size], mask)

sin, bac, env, errs, z1s = fit_thon(q_rav, 0.00035, z, wav, thon_rav)

plot = pg.plot(thon_rav[0:])
plot.plot((env*sin + bac)[0:], pen=pg.mkPen('y'))


plot = pg.plot( (thon_rav-bac)/env )
plot.plot(sin , pen=pg.mkPen('y'))
