from .utils import integrate
from .utils import Cgls
import numpy as np

def integrate_old(dssin, dfsin, maskin, maxiter=3000, stepin=[1.,1.]):
    # first rescale the problem so that values are ~1
    scale = np.mean(np.array([dssin, dfsin]))
    dss = dssin / scale
    dfs = dfsin / scale
    mask = maskin / np.mean(maskin)
    step = [1, stepin[1]/stepin[0]]
    
    def grad_ss(p):
        return (p[1:, :] - p[:-1, :])/step[0]

    def grad_fs(p):
        return (p[:, 1:] - p[:, :-1])/step[1]

    def f(x):
        return np.sum( mask * (grad_ss(x)[:,:-1]-dss)**2 ) + np.sum( mask * (grad_fs(x)[:-1,:]-dfs)**2 )

    def df(x):
        out = np.zeros_like(x)
        #
        # outi,j       = [xi,j       - xi-1,j      - dssi-1,j ] mi-1,j
        out[1:, :-1]  += ((x[1:, :-1] - x[:-1, :-1])/step[0]**2 - dss[:, :])*mask
        #
        # outi,j       = [xi+1,j     - xi,j        - dssi,j   ] mi,j
        out[:-1, :-1] -= ((x[1:, :-1] - x[:-1, :-1])/step[0]**2 - dss[:, :])*mask
        #
        # outi,j       = [xi,j       - xi,j-1      - dfsi,j-1 ] mi,j-1
        out[:-1, 1:]  += ((x[:-1, 1:] - x[:-1, :-1])/step[1]**2 - dfs[:, :])*mask
        #
        # outi,j       = [xi,j+1       - xi,j      - dfsi,j   ] mi,j
        out[:-1, :-1] -= ((x[:-1, 1:] - x[:-1, :-1])/step[1]**2 - dfs[:, :])*mask
        return 2*out 

    def dfd(x, d):
        out = 0.
        #
        # out += di,j di,j mi-1,j 
        #
        # out -= di,j di-1,j mi-1,j
        #
        # out -= di,j di+1,j mi,j
        #
        # out += di,j di,j mi,j
        #
        # out += di,j di,j mi,j-1
        #
        # out -= di,j di,j-1 mi,j-1
        #
        # out -= di,j di,j+1 mi,j
        #
        # out += di,j di,j mi,j
        out = 2*np.sum((
                    d[1:,:-1]  * d[1:,:-1]/step[0]**2  -  d[1:,:-1]  * d[:-1,:-1]/step[0]**2 \
                   -d[:-1,:-1] * d[1:,:-1]/step[0]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[0]**2 \
                   +d[:-1,1:]  * d[:-1,1:]/step[1]**2  -  d[:-1,1:]  * d[:-1,:-1]/step[1]**2 \
                   -d[:-1,:-1] * d[:-1,1:]/step[1]**2  +  d[:-1,:-1] * d[:-1,:-1]/step[1]**2
                    )*mask)
        return out
    
    #assert(np.allclose(df(phi), 0*phi))
    
    fd = lambda x, d : np.sum( d * df(x))
    
    tx = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    ty = np.zeros((dss.shape[0]+1, dss.shape[1]+1), dtype=float)
    
    tx[1:,:-1]  = step[0] * np.cumsum(dss, axis=0)
    #tx[:, 0]  = tx[:, 1] 
    #tx[:, -1] = tx[:, -2] 
    tx  = tx - tx[tx.shape[0]//2, :]
    ty[1:,:-1]  = step[1] * np.cumsum(dfs, axis=1)
    #ty[:, 0]  = ty[:, 1] 
    #ty[:, -1] = ty[:, -2] 
    ty  = (ty.T - ty[:, ty.shape[1]//2]).T
    
    x0 = (tx + ty) / 2.
    x0[0, :]  = x0[1, :]
    x0[:, -1] = x0[1, -2]
    
    cgls = Cgls(x0, f, df, fd, dfd=dfd, imax=maxiter)

    scale *= stepin[0]
    
    phase = scale * cgls.cgls()

    dss_forw = grad_ss(phase)[:,:-1]/stepin[0] 
    dfs_forw = grad_fs(phase)[:-1,:]/stepin[0]
    
    return phase[:-1, :-1], {'dss_forward': dss_forw,
                             'dfs_forward': dfs_forw,
                             'dss_residual': dss_forw-dss,
                             'dfs_residual': dfs_forw-dfs,
                             'cgls_errors': cgls.errors}

def integrate_pixel_map(pixel_map, weight, wavlength, z, zr, n0, m0, x_pixel_size, y_pixel_size, dx, dy, maxiter=3000):
    r"""
    Integrate the pixel map to produce the phase and angle profile of the lens pupil. 
    
    Parameters
    ----------

    Returns
    -------
    
    Notes
    -----
    The pixel map (:math:`u[i, j]`) defines the pixel mapping between the reference 
    and recorded images. In 1D this is given by:
    
    :math:`I[i] = O[u[i] + n_0]`
    
    In physical units this represents the mapping:
    
    .. math:: 
    
        \begin{align}
        I(x) &= O(x - \nabla \Phi'(x)) \\ 
        I[i] &= I(i \Delta_{ss}) \\
             &= O(i \Delta_{ss} - \nabla \Phi'(i \Delta_{ss})) \\
             &= O[i \frac{\Delta_{ss}}{\Delta_x} - \frac{1}{\Delta_x}\nabla \Phi'(i \Delta_{ss})] \; \text{where}\\
        O[i] &\equiv O(i \Delta_x) \quad I[i] \equiv I(i \Delta_{ss}) \; \text{and} \\
        \Phi'(x) &\equiv \frac{\lambda z}{2\pi} \Phi(x)
        \end{align}

    Therefore we have that:

    .. math:: 
    
        \begin{align}
        u[i] &= i \frac{\Delta_{ss}}{\Delta_x} - \frac{1}{\Delta_x}\nabla \Phi'(i \Delta_{ss}) - n_0 \; \text{or}\\
        \nabla \Phi'(i \Delta_{ss}) &= i \Delta_{ss} - \Delta_x(u[i] + n_0) \\
        \Phi'(i \Delta_{ss}) &\approx  \Delta_{ss} \sum_{j=0}^{j=i} \left[ j \Delta_{ss} - \Delta_x (u[j] + n_0)\right] \\
                             &= \Delta_x\Delta_{ss}(i+1) \left[\frac{i\Delta_{ss}}{2\Delta_x} - n_0 - \frac{1}{i+1} \sum_{j=0}^{j=i} u[j] \right]
        \end{align}
    
    The angle of each ray pointing from pixel i in the detector is given by 
    :math:`\theta(i\Delta_{ss}) = -\frac{1}{z}\nabla \Phi'(i\Delta_{ss}) 
    = \frac{1}{z} \left(i \Delta_{ss} - \Delta_x(u[i] + n_0)\right)`. 
    Another way to think of this is:
    
    .. math:: 
    
        \begin{align}
        \theta(i\Delta_{ss}) &= \frac{\text{position in detector}-\text{position in object}}{z} \\
                             &= \frac{i \Delta_{ss} - (u[i] + n_0)\Delta_x }{z} \\
        \end{align}
    
    The residual angles, the ray angles after subtracting the global curvature 
    :math:`\theta_r(x) = \theta(x) - \theta_0(x)`, are given by:

    .. math:: 
     
        \begin{align}
        \nabla \Phi_0'(x) &= \frac{\lambda z}{2\pi} \frac{2\pi x}{\lambda z_r} = x \frac{z}{z_r} \\
        \nabla \Phi_r'(i \Delta_{ss}) &=  i \Delta_{ss} - \Delta_x(u[i] + n_0) - i \Delta_{ss} \frac{z}{z_r} \\
                                      &=  i \Delta_{ss} \frac{z_r-z}{z_r} - \Delta_x(u[i] + n_0)  \\
        \theta_r(x) &= \frac{i \Delta_{ss} \frac{z_r - z}{z_r} - \Delta_x(u[i] + n_0)}{z}  \\
        \Phi_r'(i \Delta_{ss}) &= \Delta_{ss} \sum_{j=0}^{j=i} \left[ j\frac{z_r - z}{z_r} \Delta_{ss} - \Delta_x (u[j] + n_0)\right] 
        \end{align}
    """
    #\theta_r(x) &= \frac{i \Delta_{ss} \frac{z_r - z}{z_r} - \Delta_x(u[i] + n_0)}{z}  
    i, j = np.indices(pixel_map.shape[1:])
    theta    = np.zeros_like(pixel_map)
    theta[0] = (i * x_pixel_size * (zr - z)/zr - dx * (pixel_map[0] + n0))/z
    theta[1] = (j * y_pixel_size * (zr - z)/zr - dy * (pixel_map[1] + m0))/z
    t, res = integrate_old(
                 theta[0], theta[1], 
                 weight, maxiter, [x_pixel_size, y_pixel_size])
    #phase *= 2. * np.pi / (wavlength * z_sample_dist) 
    #return phase, {'pixel_map': np.array([res['dss_forward'], res['dfs_forward']])}
    return theta, {'pixel_map': np.array([res['dss_forward'], res['dfs_forward']])}
