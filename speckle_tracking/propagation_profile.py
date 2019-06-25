import numpy as np
import tqdm

from .utils import bilinear_interpolation_array

def propagation_profile(phase, W, z, wav, x_pixel_size, y_pixel_size, X, zs=[-1e-4, 1e4, 1000], Nint=4, axis=1):
    """
    """
    Npad = 1
    # zero padd and interpolate
    Wp, ss, fs     = zero_padd_interp(W, Npad, Nint)
    phasep, ss, fs = zero_padd_interp(phase, Npad, Nint)
    s = np.sqrt(Wp) * np.exp(1J * phasep) 
    
    # propagate then downsample
    zs = np.linspace(zs[0], zs[1], zs[2])
    rs = []

    # centre the optical axis in the array
    ss = ss - np.mean(ss)
    fs = fs - np.mean(fs)
    q = -1j * np.pi * (ss**2 * x_pixel_size**2 + fs**2 * y_pixel_size**2)/ (wav*z**2)
    qq = np.exp(q)
    
    for n in tqdm.trange(len(zs), desc='propagating'):
        t = s * np.exp(q * zs[n])
        t = np.abs(np.fft.fftn(t.astype(np.complex64)))**2
        t = np.fft.fftshift(np.sum(t, axis=1-axis))
        t = np.sum(t.reshape(t.shape[0]//Nint, Nint), axis=-1)
        rs.append(t)
    
    return np.array(rs)

def zero_padd_interp(array, Npad, Nint):
    ss = np.arange(-(Npad-1)*array.shape[0]/2, (Npad+1)*array.shape[0]/2,1/Nint)
    fs = np.arange(-(Npad-1)*array.shape[1]/2, (Npad+1)*array.shape[1]/2,1/Nint)
    
    ss, fs = np.meshgrid(ss, fs, indexing='ij')
    out = bilinear_interpolation_array(array, ss, fs, fill=0, invalid=0)
    return out, ss, fs

def downsample(I, n = 2):
    out = np.sum(I.reshape((I.shape[0], I.shape[1]//n, n)), axis=-1)
    out = np.sum(out.T.reshape((I.shape[1]//n, I.shape[0]//n, n)), axis=-1).T
    return out
