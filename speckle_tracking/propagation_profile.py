import numpy as np
import tqdm
import scipy.ndimage

from .utils import bilinear_interpolation_array

def propagation_profile(phase, W, z, wav, x_pixel_size, y_pixel_size, zs=[-1e-4, 1e-4, 1000]):
    """
    """
    # estimate beam angular half width from extent of whitefield
    X = max(np.array(W.shape) * np.array([x_pixel_size, y_pixel_size]))
    theta = np.arctan2(X/2, z)
    
    # find the required sampling
    f_fs = np.max(np.abs(np.diff(phase, axis=1)))/(2*np.pi*y_pixel_size)
    f_ss = np.max(np.abs(np.diff(phase, axis=0)))/(2*np.pi*x_pixel_size)
    dfs_max = min(1/(2*f_fs), y_pixel_size)
    dss_max = min(1/(2*f_ss), x_pixel_size)
    Npad = 3
    
    # zero padd and interpolate onto a square grid
    dss0 = dfs0 = min(dfs_max, dss_max)
    N     = max(W.shape)
    shape = (Npad*N, Npad*N)
    # 
    Wp, ss, fs     = zero_padd_interp(W,     x_pixel_size, y_pixel_size, dss0, dfs0, shape)
    phasep, ss, fs = zero_padd_interp(phase, x_pixel_size, y_pixel_size, dss0, dfs0, shape)
    
    # now filter the whitefield 
    Wp, window = window_filter(Wp, alpha=0.5, L=1)
    Wp = np.abs(Wp)
    Pmin_ss = 0.5 * dss0 
    Pmin_fs = 0.5 * dfs0 
     
    # form the wavefront in the plane of the detector
    s = np.sqrt(Wp) * np.exp(1J * phasep) 
    
    # propagate then downsample
    zs, zstep = np.linspace(zs[0], zs[1], zs[2], retstep=True)
    pss = []
    pfs = []
    
    Xp = np.max(np.abs(zs))*np.tan(theta)
    xs = 4*Xp*np.fft.fftshift(np.fft.fftfreq(len(zs)))
    
    for n in tqdm.trange(len(zs), desc='propagating'):
        t, dss, dfs = prop_divergent(s, dss0, dfs0, z, zs[n], wav, theta, Pmin_ss, Pmin_fs)
        
        t = np.abs(t)**2
        tss = np.sum(t, axis=1)
        tfs = np.sum(t, axis=0)
        
        tss = interpolate_prop(tss, dss, xs)
        tfs = interpolate_prop(tfs, dfs, xs)
        
        # interpolate tx and ty onto a regular grid
        pss.append(tss)
        pfs.append(tfs)
        
    dx = dy = xs[1]-xs[0]
    
    out = np.array([pss, pfs])
    return np.transpose(out, (0, 2, 1)), dx, dy, zstep

def propagation_profile_old(phase, W, z, wav, x_pixel_size, y_pixel_size, zs=[-1e-4, 1e-4, 1000], Nint=4):
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

def zero_padd_interp(array, dss_old, dfs_old, dss_new, dfs_new, shape_new):
    ss = dss_new / dss_old * (np.arange(shape_new[0]) - shape_new[0]/2) + array.shape[0]/2
    fs = dfs_new / dfs_old * (np.arange(shape_new[1]) - shape_new[1]/2) + array.shape[1]/2
    
    ss, fs = np.meshgrid(ss, fs, indexing='ij')
    out = bilinear_interpolation_array(array, ss, fs, fill=0, invalid=0)
    return out, ss, fs

def zero_padd_interp_old(array, Npad, Nint):
    ss = np.arange(-(Npad-1)*array.shape[0]/2, (Npad+1)*array.shape[0]/2,1/Nint)
    fs = np.arange(-(Npad-1)*array.shape[1]/2, (Npad+1)*array.shape[1]/2,1/Nint)
    
    ss, fs = np.meshgrid(ss, fs, indexing='ij')
    out = bilinear_interpolation_array(array, ss, fs, fill=0, invalid=0)
    return out, ss, fs

def downsample(I, n = 2):
    out = np.sum(I.reshape((I.shape[0], I.shape[1]//n, n)), axis=-1)
    out = np.sum(out.T.reshape((I.shape[1]//n, I.shape[0]//n, n)), axis=-1).T
    return out

def window_filter(array, alpha=0.5, L=1):
    wss = Tukey_window(array.shape[0], alpha, L)
    wfs = Tukey_window(array.shape[1], alpha, L)
    window = wss[:, np.newaxis] * wfs[np.newaxis, :]
    out  = np.fft.fftn(array.copy())
    out *= window
    return np.fft.ifftn(out).real.astype(array.dtype), window

def Tukey_window(N, alpha=0.5, L=1):
    """
    https://en.wikipedia.org/wiki/Window_function#Tukey_window
    
    with 2x zero padding 
    """
    xs  = np.abs(np.fft.fftfreq(N, 1/2))
    out = np.zeros_like(xs)
    x0 = (1-alpha) * L/2
    #
    m = xs <= x0
    out[m] = 1
    #
    m = (xs > x0) * (xs <= L/2)
    out[m] = (1+np.cos(2*np.pi*(xs[m]-x0)/(alpha * L)))/2
    return out

def prop_divergent(v, dss, dfs, z, zp, wav, theta, Pmin_ss, Pmin_fs):
    N = max(v.shape)
    dx = max(dss, dfs)
    # point at which Fresnel and Fourier sampling are equal
    z0 = z / (N*dx**2/(z*wav)-1)
    #
    if np.abs(zp) > np.abs(z0):
        return scaled_Fresnel_prop(v, dss, dfs, z, zp, wav, True, theta, Pmin_ss, Pmin_fs)
    else :
        return        Fourier_prop(v, dss, dfs, z, zp, wav, True, theta)


def scaled_Fresnel_prop(v, dss, dfs, z1, z2, wav, 
                        check_sampling=False, theta=None, 
                        Pmin_ss=None, Pmin_fs=None, normalise=True):
    """
    Frenel propagate psi a distance dz
    
    uses a tukey window to avoid aliasing
    """
    out = v.copy()
    N, M = v.shape
    dz  = z2-z1
    if Pmin_ss is None :
        Pmin_ss = dss/2
    if Pmin_fs is None :
        Pmin_fs = dfs/2
    #
    if check_sampling:
        afs = np.abs(2*wav*z1*dz/(z2*Pmin_fs))
        ass = np.abs(2*wav*z1*dz/(z2*Pmin_ss))
        b = np.abs(4*z1*np.tan(theta))
        if (N*dss) < min(ass, b):
            print("\nscaled_Fresnel_prop")
            print("Warning: increase N or dss")
            print("N*dy:", N*dss)
            print("np.abs(2*wav*z1*dz/(z2*Pmin_ss))", ass)
            print("np.abs(4*z1*np.tan(theta))", b)
        if (M*dfs) < min(afs, b):
            print("\nscaled_Fresnel_prop")
            print("Warning: increase M or dfs")
            print("M*dx:", M*dfs)
            print("np.abs(2*wav*z1*dz/(z2*Pmin_fs))", afs)
            print("np.abs(4*z1*np.tan(theta))", b)
    #
    dq_fs = z1/(M*dss)
    dq_ss = z1/(N*dfs)
    qfs = dq_fs * np.fft.fftfreq(M, 1/M)
    qss = dq_ss * np.fft.fftfreq(N, 1/N)
    q2 = (qss**2)[:, np.newaxis] + (qfs**2)[np.newaxis, :]
    out = np.fft.fftn(out)
    exp = np.exp(-1J * np.pi * wav * dz/(z1*z2) * q2)
    out = np.fft.ifftn(out*exp)
    if z2 < 0 :
        out = out[::-1, ::-1]

    dss_out = np.abs(z2/z1 * dss)
    dfs_out = np.abs(z2/z1 * dfs)
    if normalise is True :
        n_in  = np.abs(dss*dfs)*np.sum(np.abs(v)**2)
        n_out = dss_out*dfs_out*np.sum(np.abs(out)**2)
        out *= np.sqrt(n_in / n_out)
    return out, dss_out, dfs_out

def Fourier_prop(v, dss, dfs, z1, z2, wav, check_sampling=False, theta=None, normalise=True):
    """
    u(x, z_1) = e^{i\pi x^2/(\lambda z_1)} v(x, z_1)
    
    u(x, z_2) = e^{i\pi x^2/(\lambda \Delta z)} / \sqrt{-i} 
      \times F[e^{i\pi x^2/\lambda  z_2/(z_1 \Delta z) v(x, z_1)](q=x/\lambda \Delta z)
    """
    N, M = v.shape
    out  = np.fft.ifftshift(v)
    dz   = z2-z1
    #
    if check_sampling:
        #assert((dx*N)>(4*z1*np.tan(theta)))
        if (dss*N)<(4*z1*np.tan(theta)):
            print("\nFourier_prop")
            print("Warning: increase N or dss")
            print("N*dss:", N*dss)
            print("4*z1*np.tan(theta):", 4*z1*np.tan(theta))
        if (dfs*M)<(4*z1*np.tan(theta)):
            print("\nFourier_prop")
            print("Warning: increase M or dfs")
            print("M*dfs:", M*dfs)
            print("4*z1*np.tan(theta):", 4*z1*np.tan(theta))
        #assert(dx < (wav * dz / (2*z2*np.tan(theta))))
        if np.abs(z2)>0:
            if dss>np.abs(wav * dz / (2*z2*np.tan(theta))):
                print("Warning: decrease dss")
                print("dss:", dss)
                print("np.abs(wav * dz / (2*z2*np.tan(theta))):", np.abs(wav * dz / (2*z2*np.tan(theta))))
            if dfs>np.abs(wav * dz / (2*z2*np.tan(theta))):
                print("Warning: decrease dfs")
                print("dfs:", dfs)
                print("np.abs(wav * dz / (2*z2*np.tan(theta))):", np.abs(wav * dz / (2*z2*np.tan(theta))))
        
    ss   = dss*np.fft.fftfreq(N, 1/N)
    fs   = dfs*np.fft.fftfreq(M, 1/M)
    x2   = ((ss**2)[:, np.newaxis] + (fs**2)[np.newaxis, :])
    exp  = np.exp(1J *np.pi* x2 * z2 / (wav*z1*dz))
    out  = np.fft.ifftn(exp*out, norm="ortho")
    dss2  = -wav * dz/(N*dss)
    dfs2  = -wav * dz/(M*dfs)
    ss   = dss2*np.fft.fftfreq(N, 1/N)
    fs   = dfs2*np.fft.fftfreq(M, 1/M)
    x2   = ((ss**2)[:, np.newaxis] + (fs**2)[np.newaxis, :])
    out *= np.exp(1J * np.pi * x2**2 / (wav * dz)) 
    if np.abs(z2) > 0 :
        out *= np.exp(-1J * np.pi * x2**2 / (wav * z2)) 
    out /= np.sqrt(-1J)

    if normalise is True :
        n_in  = np.abs(dss*dfs)*np.sum(np.abs(v)**2)
        n_out = np.abs(dss2*dfs2)*np.sum(np.abs(out)**2)
        out *= np.sqrt(n_in / n_out)
    return np.fft.fftshift(out), dss2, dfs2

def interpolate_prop(I, dx, xs):
    """
    assume I is centred
    """
    dx_out = xs[1] - xs[0]
    
    # filter the intensity to the desired grid
    If = scipy.ndimage.gaussian_filter1d(I, dx_out/dx, mode='constant')
    
    # now sample at the xs 
    xs_in = dx*np.fft.fftshift(np.fft.fftfreq(I.shape[0], 1/I.shape[0]))
    Iout = np.interp(xs, xs_in, If, left=0, right=0)
    
    return Iout
