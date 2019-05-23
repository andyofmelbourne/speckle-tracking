import numpy as np
import tqdm

def calc_error(data, mask, W, dij_n, I, pixel_map, n0, m0, subpixel=False, verbose=True):
    r"""
    Parameters
    ----------
    data : ndarray
        Input data, of shape (N, M, L).
    
    mask : ndarray
        Boolean array of shape (M, L), where True indicates a good
        pixel and False a bad pixel.
    
    W : ndarray
        Float array of shape (M, L) containing the estimated whitefield.
    
    dij_n : ndarray
        Float array of shape (N, 2) containing the object translations 
        that have been mapped onto the detector's frame of reference.     

    I : ndarray
        Float array of shape (U, V), this is essentially an object map. 
    
    pixel_map : ndarray, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]
    
    n0 : float
        Slow scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \ge -0.5 
            \quad\text{for all } i,j
    
    m0 : float
        Fast scan offset to the pixel mapping such that:
            
        .. math::
            
            \text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0 \ge -0.5 
            \quad\text{for all } i,j
    
    subpixel : bool, optional
        If True then use bilinear subpixel interpolation non-integer pixel mappings.
    
    verbose : bool, optional
        print what I'm doing.
    
    minimum_overlap : float or None, optional
        Default is None. If float then the the object will be set to -1 
        where the number of data points contributing to that value is less
        than "minimum_overlap".

    Returns
    -------
    
    error_total : float
        The global error value, :math:`\varepsilon = \sum_{n,i,j} \varepsilon[n, i, j]`.
    
    error_frame : ndarray
        Float array of shape (N,). The average pixel error per detector frame, 
        :math:`\varepsilon_\text{frame}[n] = \langle \varepsilon[n, i, j] \rangle_{i,j}`.
        
    error_pixel : ndarray
        Float array of shape (M, L). The average pixel error per detector pixel, 
        :math:`\varepsilon_\text{pixel}[i, j] = \langle \varepsilon[n, i, j]\rangle_{n}`.

    Notes
    -----
    The error, per pixel and per frame, is given by:

    .. math::

        \begin{align}
        \varepsilon[n, i, j] = M[i,j] \bigg[ I_\Phi[n, i, j] - 
            W[i, j] I_0[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                        &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]\bigg]^2
        \end{align}
    """
    # mask the pixel mapping
    ij     = np.array([pixel_map[0][mask], pixel_map[1][mask]])
    
    error_total = 0.
    error_frame = np.zeros(data.shape[0])
    error_pixel = np.zeros(data.shape[1:])
    norm        = np.zeros(data.shape[1:])

    for n in tqdm.trange(data.shape[0], desc='calculating errors'):
        if subpixel: 
            # define the coordinate mapping and round to int
            ss = pixel_map[0] - dij_n[n, 0] + n0
            fs = pixel_map[1] - dij_n[n, 1] + m0
            #
            I0 = W * bilinear_interpolation_array(I, ss, fs, fill=-1, invalid=-1)
            I0 = I0[mask]
        
        else :
            # define the coordinate mapping and round to int
            ss = np.rint((ij[0] - dij_n[n, 0] + n0)).astype(np.int)
            fs = np.rint((ij[1] - dij_n[n, 1] + m0)).astype(np.int)
            #
            I0 = I[ss, fs] * W[mask]
        
        d  = data[n][mask]
        m  = (I0>0)*(d>0)
        
        error_map = m*(I0 - d)**2
        tot       = np.sum(error_map)
        
        error_total       += tot
        error_pixel[mask] += error_map
        error_frame[n]     = tot / np.sum(m)
        norm[mask]        += m
    
    m = norm>0
    error_pixel[m] = error_pixel[m] / norm[m]
    return error_total, error_frame, error_pixel

def bilinear_interpolation_array(array, ss, fs, fill = -1, invalid=-1):
    """
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    out = np.zeros(ss.shape)
    
    s0, s1 = np.floor(ss).astype(np.uint32), np.ceil(ss).astype(np.uint32)
    f0, f1 = np.floor(fs).astype(np.uint32), np.ceil(fs).astype(np.uint32)
    
    # check out of bounds
    m = (ss > 0) * (ss <= (array.shape[0]-1)) * (fs > 0) * (fs <= (array.shape[1]-1))

    s0[~m] = 0
    s1[~m] = 0
    f0[~m] = 0
    f1[~m] = 0
    
    # careful with edges
    s1[(s1==s0)*(s0==0)] += 1
    s0[(s1==s0)*(s0!=0)] -= 1
    f1[(f1==f0)*(f0==0)] += 1
    f0[(f1==f0)*(f0!=0)] -= 1
    
    # make the weighting function
    w00 = (s1-ss)*(f1-fs)
    w01 = (s1-ss)*(fs-f0)
    w10 = (ss-s0)*(f1-fs)
    w11 = (ss-s0)*(fs-f0)
    
    # renormalise for invalid pixels
    w00[array[s0,f0]==invalid] = 0.
    w01[array[s0,f1]==invalid] = 0.
    w10[array[s1,f0]==invalid] = 0.
    w11[array[s1,f1]==invalid] = 0.
    
    # if all pixels are invalid then return fill
    s = w00+w10+w01+w11
    m = (s!=0)*m
    
    out[m] = w00[m] * array[s0[m],f0[m]] \
           + w10[m] * array[s1[m],f0[m]] \
           + w01[m] * array[s0[m],f1[m]] \
           + w11[m] * array[s1[m],f1[m]]
    
    out[m] /= s[m]
    out[~m] = fill
    return out  

def bilinear_interpolation(array, ss, fs, fill = -1, invalid=-1):
    """
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    """
    import math
    # check out of bounds
    if (ss < 0) or (ss> (array.shape[0]-1)) or (fs < 0) or (fs > (array.shape[1]-1)):
        return fill
    
    s0, s1 = math.floor(ss), math.ceil(ss)
    f0, f1 = math.floor(fs), math.ceil(fs)
    
    # careful with edges
    if s1==s0 :
        if s0 == 0 :
            s1 += 1
        else :
            s0 -= 1
    
    if f1==f0 :
        if f0 == 0 :
            f1 += 1
        else :
            f0 -= 1

    # make the weighting function
    w = np.zeros((2,2), dtype=float)
    a = np.array([[array[s0, f0], array[s0, f1]],
                   array[s1, f0], array[s1, f1]])
    w[0, 0] = (s1-ss)*(f1-fs)
    w[0, 1] = (s1-ss)*(fs-f0)
    w[1, 0] = (ss-s0)*(f1-fs)
    w[1, 1] = (ss-s0)*(fs-f0)
    
    # renormalise for invalid pixels
    w[a==invalid] = 0.
    s = np.sum(w)

    # if all pixels are invalid then return fill
    if s == 0 :
        return fill
    
    w = w/np.sum(w)
    return np.sum( w * a )
