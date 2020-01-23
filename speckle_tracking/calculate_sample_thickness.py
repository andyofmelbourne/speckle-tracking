import numpy as np

def calculate_sample_thickness( delta, beta, zt, defocus, wav, dss, dfs, reference_image, roi_ref, set_median_to_zero=True, tol_ctf=2e-1, tol_tie=1e-2):
    """
    deltas and betas from: http://henke.lbl.gov/optical_constants/getdb2.html
    """
    # cut the reference image to reduce noise
    roi = (slice(roi_ref[0], roi_ref[1]), slice(roi_ref[2], roi_ref[3]))
    O = reference_image[roi]
    
    # calculate the effective defocus distance
    z1 = defocus
    z = zt-z1
    ze = z*z1/(z1+z)
    
    t_pag = paganin( delta, beta, ze, wav, dss, dfs, O, tol=tol_tie)
    t_ctf = CTF_inversion( delta, beta, ze, wav, dss, dfs, O, tol=tol_ctf)
    
    # set the substrate to 0
    if set_median_to_zero :
        t_pag -= np.median(t_pag)
        t_ctf -= np.median(t_ctf)
    
    # un-roi the profiles
    out_pag = np.zeros_like(reference_image)
    out_ctf = np.zeros_like(reference_image)

    out_pag[roi] = t_pag
    out_ctf[roi] = t_ctf
    return out_pag, out_ctf


def paganin( delta, beta, z, wav, dss, dfs, Iin, tol=1e-6) :
    """function Uin= paganin( Iout, delta, beta, distance,lambda, dx, Iin)
    This function retrieves the object thickness map (Uin) with certain delta and beta
    refractieve properties, given a progagated Intensity (Iout)
    characterised by lambda at a certain distance, sampled with dx
    spacing.
    
    see: https://arxiv.org/pdf/1902.00364.pdf eq 61
    """
    N, M = Iin.shape
    fr = np.fft.fftfreq(N, d=dss)
    fc = np.fft.fftfreq(M, d=dfs)
    fx, fy = np.meshgrid(fr,fc, indexing='ij')
    f2 = (fx**2 + fy**2) # ?
    
    k  = 2*np.pi/wav
    mu = 2*k*beta
    
    If = np.fft.fftn(Iin)
    
    #Iout =  (If / (Iin+tol)) / (1 + (delta * z / mu) * f2)
    Iout =  (If / np.mean(Iin)) / (tol + (delta * z / mu) * f2)
    Iout =  -1/mu * np.log(np.fft.ifftn(Iout).real)
    return Iout

def CTF_inversion( delta, beta, z, wav, dss, dfs, Iin, tol=0.5) :
    N, M   = Iin.shape
    fr     = np.fft.fftfreq(N, d=dss)
    fc     = np.fft.fftfreq(M, d=dfs)
    fx, fy = np.meshgrid(fr,fc, indexing='ij')
    f2     = np.pi*(fx**2 + fy**2)
    q      = np.pi*np.linspace(0, f2.max(), 1000)
    
    d = np.sin(wav * z * f2) + beta / delta * np.cos(wav * z * f2)
    dmax = 1/tol
    m   = np.abs(d)<(1/dmax) 
    d[m] = np.sign(d[m]) / dmax

    out = np.fft.fftn(Iin)
    out = out/(2*d*delta)
    return -np.fft.ifftn(out).real * wav / (2. * np.pi)
