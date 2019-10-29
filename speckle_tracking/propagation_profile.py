import numpy as np
import tqdm

from .utils import bilinear_interpolation_array

def propagation_profile(phase, W, z, wav, x_pixel_size, y_pixel_size, X, zs=[-1e-4, 1e-4, 1000], Nint=4, axis=1):
    """
    """
    Npad = 1
    # zero padd and interpolate
    Wp, ss, fs     = zero_padd_interp(W, Npad, Nint)
    phasep, ss, fs = zero_padd_interp(phase, Npad, Nint)
    s = np.sqrt(Wp) * np.exp(1J * phasep) 
    
    # propagate then downsample
    zs, zstep = np.linspace(zs[0], zs[1], zs[2], retstep=True)
    px = []
    py = []
    
    # centre the optical axis in the array
    ss = ss - np.mean(ss)
    fs = fs - np.mean(fs)
    
    # now make the z-step propagator
    # dx = 1 / Q = 1 / (N du / wav z) = wav z / (N du)
    # i pi x^2 / wav z = i pi (n wav z / (N du))**2 / wav dz
    #                  = i pi wav (n z / (N du))**2 / dz
    #ex = np.exp(1j * np.pi * wav * (ss * z / (ss.shape[0] * x_pixel_size))**2 / zstep)
    #ex = np.exp(1j * np.pi * wav * (ss * z / (ss.shape[0] * x_pixel_size))**2 / zstep)
    q = -1j * np.pi * (ss**2 * x_pixel_size**2 + fs**2 * y_pixel_size**2)/ (wav*z**2)
    ex = np.exp(zstep * q)
    
    s2 = s * np.exp(q * zs[0])
    tt = zs[0]
    for n in tqdm.trange(len(zs), desc='propagating'):
        t = np.fft.ifftn(s2)
        t = np.abs(t)**2
        tx = np.fft.fftshift(np.sum(t, axis=1))
        tx = np.sum(tx.reshape(tx.shape[0]//Nint, Nint), axis=-1)
        ty = np.fft.fftshift(np.sum(t, axis=0))
        ty = np.sum(ty.reshape(ty.shape[0]//Nint, Nint), axis=-1)
        px.append(tx)
        py.append(ty)
        
        s2 *= ex

    px = np.array(px)
    py = np.array(py)

    dx = wav * z / (phase.shape[0]*x_pixel_size)
    dy = wav * z / (phase.shape[1]*y_pixel_size)
    return px, py, dx, dy, zstep

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
