import speckle_tracking as st
import h5py
import numpy as np
import pyqtgraph as pg
import scipy.signal


def make_thon(data, mask, W, roi=None):
    sig = data.shape[-1]//8
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

def fit_envolopes(q, z1, zD, wav, thon_rav):
    pass


def interp(xn, xo, y):
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
        im[im<0] = 0
        im[im>=len(y)] = len(y)-1
        i[im<0] = 0
        i[im>=len(y)] = len(y)-1
        
        out = (xo[i]-xn)*y[im] + (xn-xo[im])*y[i]
        d   = xo[i]-xo[im]
        out[d>0] = out[d>0]/d[d>0]
   return out


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

thon = make_thon(data, mask, W, roi)

q, t = get_r_theta(thon.shape, [x_pixel_size, y_pixel_size], False)

r     = q * np.sqrt(thon.shape[0]**2 + thon.shape[1]**2)/2 / q.max()
thon_rav = st.radial_symetry(thon, r, mask)
q_rav    = np.linspace(0, q.max(), thon_rav.shape[0])

#var, rav = find_symmetry(np.fft.ifftshift(thon)**0.1)

#f = h5py.File('siemens_star.cxi', 'a')
#f['results/mask'] = mask
#f.close()
