import numpy as np
import tqdm

def make_object_map(data, mask, W, dij_n, pixel_map, subpixel=False, verbose=True, minimum_overlap=None):
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
    
    pixel_map : ndarray, (2, M, L)
        An array containing the pixel mapping 
        between a detector frame and the object, such that: 
        
        .. math:: 
        
            I^{z_1}_{\phi}[n, i, j]
            = W[i, j] I^\infty[&\text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0,\\
                               &\text{ij}_\text{map}[1, i, j] - \Delta ij[n, 1] + m_0]

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
    I : ndarray
        Float array of shape (U, V), this is essentially an object map given by:
        
        .. math::

            I_0[i, j] &= 
            \frac{\sum_n M[u_n, v_n] W[u_n, v_n] I^{z_1}_\phi[n, u_n, v_n]}{\sum_n M[u_n, v_n] W[u_n, v_n]^2 } \\
        
        where: 
        
        .. math::

            \begin{align}
            u_n[i,j] &= \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \\
            u_n[i,j] &= \text{ij}_\text{map}[0, i, j] - \Delta ij[n, 0] + n_0 \\
            \end{align}
        
        see Notes for more.
    
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
        
        -0.5 is chosen rather than 0 because integer coordinates are defined 
        at the centre of the physical pixel locations.
    
    Notes
    -----
    .. math::
        
        I_{\phi, n}(\mathbf{x})
        = W(\mathbf{x})I_0(\mathbf{x} - \frac{\lambda z}{2\pi} \nabla \Phi(x)-\Delta x_n, \bar{z}_\Phi)
    
    :math:`M, W` are the mask and whitefield arrays respectively. 
    U and V are the pixel dimensions of :math:`I_0` given by:
     
    .. math::
        
        U &= \text{max}(\text{ij}_\text{map}[0, i, j]) - \text{min}(\Delta ij_n[0]) + n_0 \\
        V &= \text{max}(\text{ij}_\text{map}[1, i, j]) - \text{min}(\Delta ij_n[1]) + m_0
    """
    # mask the pixel mapping
    ij     = np.array([pixel_map[0][mask], pixel_map[1][mask]])
    
    # choose the offset so that ij - dij_n + n0 > -0.5
    # for all unmasked pixels
    n0, m0 = -np.min(ij[0]) + np.max(dij_n[:, 0]), -np.min(ij[1]) + np.max(dij_n[:, 1])
    n0, m0 = n0-0.5, m0-0.5
    
    # determine the object-map domain
    shape   = (int(np.rint(np.max(ij[0]) - np.min(dij_n[:, 0]) + n0))+1, \
               int(np.rint(np.max(ij[1]) - np.min(dij_n[:, 1]) + m0))+1)
    I       = np.zeros(shape, dtype=np.float)
    overlap = np.zeros(shape, dtype=np.float)
    WW      = W**2
    
    if subpixel :
        for n in tqdm.trange(data.shape[0], desc='building object map'):
            # define the coordinate mapping
            ss = pixel_map[0] - dij_n[n, 0] + n0
            fs = pixel_map[1] - dij_n[n, 1] + m0
            
            I = bilinear_interpolation_array_inverse(
                                           I, W*data[n], ss, fs, invalid = mask)
            
            overlap = bilinear_interpolation_array_inverse(
                                           overlap, WW, ss, fs, invalid = mask)
        
    else :
        for n in tqdm.trange(data.shape[0], desc='building object map'):
            # define the coordinate mapping and round to int
            ss = np.rint((ij[0] - dij_n[n, 0] + n0)).astype(np.int)
            fs = np.rint((ij[1] - dij_n[n, 1] + m0)).astype(np.int)
            #
            I[      ss, fs] += (W*data[n])[mask]
            overlap[ss, fs] += WW[mask]
            
    overlap[overlap<1e-2] = -1
    m = (overlap > 0)
    
    I[m]  = I[m] / overlap[m]
    I[~m] = -1
    
    if minimum_overlap is not None :
        m = overlap < minimum_overlap
        I[m] = -1
    
    return I, n0, m0

def bilinear_interpolation_array_inverse(out, array, ss, fs, invalid=-1):
    """
    See https://en.wikipedia.org/wiki/Bilinear_interpolation
    
    out[ss[i, j], fs[i, j]] += array[i, j]
    """
    s0, s1 = np.floor(ss).astype(np.uint32), np.ceil(ss).astype(np.uint32)
    f0, f1 = np.floor(fs).astype(np.uint32), np.ceil(fs).astype(np.uint32)
    
    # check out of bounds or invalid pixels
    m = (ss > 0) * (ss <= (out.shape[0]-1)) * (fs > 0) * (fs <= (out.shape[1]-1))
    
    if type(invalid) is np.ndarray :
        m = m * invalid
    else :
        m = m * (array!=invalid)
    
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
    w00 = (s1-ss)*(f1-fs)*array
    w01 = (s1-ss)*(fs-f0)*array
    w10 = (ss-s0)*(f1-fs)*array
    w11 = (ss-s0)*(fs-f0)*array
    
    out[s0[m],f0[m]] += w00[m]
    out[s1[m],f0[m]] += w10[m]
    out[s0[m],f1[m]] += w01[m]
    out[s1[m],f1[m]] += w11[m]
    
    return out  
