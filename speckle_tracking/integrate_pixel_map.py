from .utils import integrate
from .utils import Cgls
import numpy as np

def integrate_pixel_map(pixel_map, weight, wavelength, z, zr, 
                        x_pixel_size, y_pixel_size, dx, dy, 
                        remove_astigmatism = False, maxiter=3000):
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
    = \frac{1}{z} \left(\Delta_x(u[i] + n_0) - i \Delta_{ss}\right)`. 
    Another way to think of this is:
    
    .. math:: 
    
        \begin{align}
        \theta(i\Delta_{ss}) &= \frac{\text{position in object}-\text{position in detector}}{z} \\
                             &= \frac{(u[i] + n_0)\Delta_x - i \Delta_{ss}}{z} \\
        \end{align}
    
    The residual angles, the ray angles after subtracting the global curvature 
    :math:`\theta_r(x) = \theta(x) - \theta_0(x)`, are given by:

    .. math:: 
     
        \begin{align}
        \nabla \Phi_0'(x) &= \frac{\lambda z}{2\pi} \frac{2\pi x}{\lambda z_r} = x \frac{z}{z_r} \\
        \nabla \Phi_r'(i \Delta_{ss}) &=  i \Delta_{ss} - \Delta_x(u[i] + n_0) - i \Delta_{ss} \frac{z}{z_r} \\
                                      &=  i \Delta_{ss} \frac{z_r-z}{z_r} - \Delta_x(u[i] + n_0)  \\
        \theta_r(x) &= \frac{\Delta_x(u[i] + n_0) - i \Delta_{ss} \frac{z_r - z}{z_r}}{z}  \\
        \Phi_r'(i \Delta_{ss}) &= \Delta_{ss} \sum_{j=0}^{j=i} \left[ j\frac{z_r - z}{z_r} \Delta_{ss} - \Delta_x (u[j] + n_0)\right] 
        \end{align}
    """
    # for now :
    n0 = m0 = 0
    #\theta_r(x) &= \frac{i \Delta_{ss} \frac{z_r - z}{z_r} - \Delta_x(u[i] + n_0)}{z}  
    i, j = np.indices(pixel_map.shape[1:])
    theta_r    = np.zeros_like(pixel_map)
    theta_r[0] = (i * x_pixel_size * (zr - z)/zr - dx * (pixel_map[0] + n0))/z
    theta_r[1] = (j * y_pixel_size * (zr - z)/zr - dy * (pixel_map[1] + m0))/z

    # remove offset and tilt
    if remove_astigmatism :
        for k in range(3): # TODO
            theta_r[0], x = remove_grad_const(theta_r[0], weight, i, 100)
            theta_r[1], x = remove_grad_const(theta_r[1], weight, j, 100)
    else :
        for k in range(3): # TODO
            theta_r, x = remove_offset_tilt_defocus(theta_r, weight, 100)
        print(x, np.mean(theta_r, axis=(1,2)))
    
    t, res = integrate(
                 theta_r[0], theta_r[1], 
                 weight, maxiter, [x_pixel_size, y_pixel_size])
    phase_r = 2 * np.pi / wavelength * t
    
    # now form the global phase and angles
    return phase_r, theta_r, {'angles_forward': np.array([res['dss_forward'], res['dfs_forward']])}

def remove_grad_const(arrayin, weight, i, maxiter):
    # x = [c, d]
    #scale = np.mean(np.sqrt(weight) * arrayin)
    scale = 1 #np.mean(np.sqrt(weight)*anglein)
    array = arrayin / scale
    def f(x):
        return np.sum( weight * (x[1] * i + x[0] - array)**2 )
    
    def df(x):
        out  = np.zeros_like(x)
        a = x[1] * i + x[0] - array
        out[0] = np.sum(weight * a)
        out[1] = np.sum(weight * i * a)
        return 2*out 
    
    fd = lambda x, d : np.sum( d * df(x))
    
    ddf = np.zeros((2,2), dtype=float)
    ddf[0,0] = np.sum(weight)
    ddf[1,1] = np.sum(weight*i**2)
    ddf[1,0] = np.sum(weight*i)
    ddf[0,1] = ddf[1,0]
    ddf = 2 * ddf 
    
    dfd = lambda x, d : np.dot(d, np.dot(ddf, d))
    
    x0 = np.array([0, 0], dtype=np.float)
    
    cgls = Cgls(x0, f, df, fd, dfd=dfd, imax=maxiter)
    
    x = cgls.cgls()
    
    out = scale*(array - x[1] * i - x[0])
    return out, x

def remove_offset_tilt_defocus(anglein, weight, maxiter):
    """
    cx / cy are position offsets (tilt)
    d is a defocus offset 
    t'x = tx - cx - d x 
    t'y = ty - cy - d y 
    """
    scale = 1 #np.mean(np.sqrt(weight)*anglein)
    angle = anglein / scale
    
    i, j = np.indices(angle.shape[1:])
    # x = [cx, cy, d]
    def f(x):
        return np.sum( weight * (x[2] * i + x[0] - angle[0])**2 + (x[2] * j + x[1] - angle[1])**2)
    
    def df(x):
        out  = np.zeros_like(x)
        a = x[2] * i + x[0] - angle[0]
        b = x[2] * j + x[1] - angle[1]
        out[0] = np.sum(a * weight)
        out[1] = np.sum(b * weight)
        out[2] = np.sum((i * a + j * b) * weight)
        return 2*out 

    fd = lambda x, d : np.sum( d * df(x))
    
    ddf = np.zeros((3,3), dtype=float)
    ddf[0,0] = np.sum(weight)
    ddf[1,1] = np.sum(weight)
    ddf[2,2] = np.sum(weight * (i**2 + j**2))
    ddf[2,0] = np.sum(weight * i)
    ddf[0,2] = ddf[2,0]
    ddf[2,1] = np.sum(weight * j)
    ddf[1,2] = ddf[2,1]
    ddf = 2 * ddf
    
    dfd = lambda x, d : np.dot(d, np.dot(ddf, d))
    
    x0 = np.array([1, 1, 1], dtype=np.float)
    
    cgls = Cgls(x0, f, df, fd, dfd=dfd, imax=maxiter)
    
    x = cgls.cgls()
    
    out = angle.copy()
    out[0] = angle[0] - x[2] * i - x[0]
    out[1] = angle[1] - x[2] * j - x[1]
    out *= scale
    return out, x

def get_defocus_old(pixel_map, weight, zt, zs, x_pixel_size, y_pixel_size, dx, dy):
    """
    fit the sample
    """
    i, j = np.indices(pixel_map.shape[1:])
    
    theta    = np.zeros_like(pixel_map)
    theta[0] = -i * x_pixel_size / zs + dx * (pixel_map[0])/zs
    theta[1] = -j * y_pixel_size / zs + dy * (pixel_map[1])/zs
    
    # now a linear fit to theta[0] or [1] will reveal the ss/fs defocus
    theta[0], x_ss = remove_grad_const(theta[0], weight, i, 100)
    theta[1], x_fs = remove_grad_const(theta[1], weight, j, 100)

    # theta = - grad phi / z = - x / z_ss = - i delta_ss / z_ss = m i 
    # z_ss  = - delta_ss / m
    z_ss = - x_pixel_size / x_ss[1]
    z_fs = - y_pixel_size / x_fs[1]
    dz = (z_ss - z_fs) / 2

    Mss, Mfs = z_ss/(z_ss-zs), z_fs/(z_fs-zs)
    
    zb = zs/2. * (1/Mss + 1/Mfs)
    zess = zs * (1/Mss)
    zefs = zs * (1/Mfs)
    
    if True : 
        print('defocus (focus-sample dist.): {:.3e}'.format(zt-zs))
        print('sample-detector dist.       : {:.3e}'.format(zs))
        print('defocus (slow scan axis)    : {:.3e}'.format(z_ss-zs))
        print('defocus (fast scan axis)    : {:.3e}'.format(z_fs-zs))
        print('Magnification       : {:.3e} (ss) {:.3e} (fs) {:.3e} (av.)'.format(Mss, Mfs, (Mss+Mfs)/2.))
        print('Effective pixel size: {:.3e}m (ss) {:.3e}m (fs) {:.3e}m (av.)'.format(x_pixel_size/Mss, y_pixel_size/Mfs, (x_pixel_size/Mss + y_pixel_size/Mfs)/2.))
        print('Effective defocus   : {:.3e}m (ss) {:.3e}m (fs) {:.3e}m (av.)'.format(zess, zefs, zb))
    return z_ss, z_fs, dz

def get_defocus(pixel_map, weight, zt, x_pixel_size, y_pixel_size, dx, dy):
    """
    fit the

    grad t = wav zs / 2 pi grad phase = i delta_ss - delta_x (pixel_map[0] + n0)
    i delta_ss zs / zx = i delta_ss - delta_x (pixel_map[0] + n0)
    i delta_fs zs / zy = i delta_fs - delta_y (pixel_map[1] + m0)

    zx = focus -> detector along ss
    zy = focus -> detector along fs
    """
    i, j = np.indices(pixel_map.shape[1:])
    
    t    = np.zeros_like(pixel_map)
    t[0] = i * x_pixel_size - dx * pixel_map[0] 
    t[1] = j * y_pixel_size - dy * pixel_map[1] 
    
    # now a linear fit to t[0] or [1] will reveal the ss/fs defocus
    a, x_ss = remove_grad_const(t[0], weight, i, 100)
    b, x_fs = remove_grad_const(t[1], weight, j, 100)
    
    mss = x_ss[1] / x_pixel_size
    mfs = x_fs[1] / y_pixel_size
    
    # sample -> detector
    zs = 2*zt / (1/mss + 1/mfs)
    
    z_ss = zs / mss 
    z_fs = zs / mfs 
    
    dz = (z_ss - z_fs) / 2
    
    # ss/fs focus -> sample 
    dz_ss = (zt-zs) + dz
    dz_fs = (zt-zs) - dz
    
    Mss, Mfs = z_ss/dz_ss, z_fs/dz_fs
    
    zb = zs/2. * (1/Mss + 1/Mfs)
    zess = zs * (1/Mss)
    zefs = zs * (1/Mfs)
    
    if True : 
        print('defocus (focus-sample dist.): {:.3e}'.format(zt-zs))
        print('sample-detector dist.       : {:.3e}'.format(zs))
        print('defocus (slow scan axis)    : {:.3e}'.format(dz_ss))
        print('defocus (fast scan axis)    : {:.3e}'.format(dz_fs))
        print('Magnification       : {:.3e} (ss) {:.3e} (fs) {:.3e} (av.)'.format(Mss, Mfs, (Mss+Mfs)/2.))
        print('Effective pixel size: {:.3e}m (ss) {:.3e}m (fs) {:.3e}m (av.)'.format(x_pixel_size/Mss, y_pixel_size/Mfs, (x_pixel_size/Mss + y_pixel_size/Mfs)/2.))
        print('Effective defocus   : {:.3e}m (ss) {:.3e}m (fs) {:.3e}m (av.)'.format(zess, zefs, zb))
    return z_ss, z_fs, dz
