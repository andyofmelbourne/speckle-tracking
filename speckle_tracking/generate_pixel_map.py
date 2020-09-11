import numpy as np

def generate_pixel_map(shape, translations, basis, x_pixel_size, 
        y_pixel_size, z, defocus_fs, defocus_ss=None, dss=None, dfs=None, verbose=True): 
    """
    Generate the pixel mapping based on the imaging geometry.
    """
    """
    zx  = x-focus to detector
    z1x = x-focus to sample
    z   = av-focus to detector
    z1  = av-focus to detector
    
    u = x - (wav (z-z1))/2 pi grad phi
    
    u[0, i, j] = u_x(i x_pixel_size, j y_pixel_size) / dx
    u[1, i, j] = u_y(i x_pixel_size, j y_pixel_size) / dy

    M = z / z_1
    dx = x_pixel_size / M
       = z_1 x_pixel_size / z

    define z1 = (z1x + z1y) / 2 
           zx = z - z1 + z1x
              = z + (z1x - z1y) / 2
     
    u_x(x, y) = x - wav (z-z1) / 2pi  2 pi x / (wav zx)
              = x[1 - (z-z1) / zx]
              = x [zx - z + z1] / zx
              = x z1x / zx
              = x / Mx
    
    u[0, i, j] = i x_pix_size / Mx / dx
    
    for dx = x_pixel_size / M :
        u[0, i, j] = i x_pix_size z1x / zx   z / (z1 x_pix_size)
                   = i z1x / zx  z / z1 
                   = i M / Mx
    
    Mx = zx / z1x
    """
    # assume no astigmatism unless otherwise stated
    if defocus_ss is None :
        defocus_ss = defocus_fs
    
    # set the average focus to sample distance and magnification
    defocus = (defocus_ss + defocus_fs) / 2.
    M       = z / defocus
    Mss     = z / defocus_ss
    Mfs     = z / defocus_fs
    
    # unless otherwise stated set: dss = dfs = demagnified pixel size
    if dss is None :
        dss = x_pixel_size / M
    
    if dfs is None :
        dfs = y_pixel_size / M
    
    # set the pixel mapping
    u = np.zeros((2,)+shape, dtype=np.float)
    i, j = np.indices(shape)
    u[0] = i * x_pixel_size / Mss / dss
    u[1] = j * y_pixel_size / Mfs / dfs
    
    # now map the sample translations to pixel units
    pixel_translations = make_pixel_translations(translations, basis, dss, dfs, x_pixel_size, y_pixel_size, verbose=verbose)
    
    return u, pixel_translations, {'dss': dss, 'dfs': dfs, 'magnification': M, 'magnification_ss': Mss, 'magnification_fs': Mfs }



def make_pixel_translations(translations, basis, dx, dy, x_pixel_size, y_pixel_size, verbose=True):
    """
    Convert sample translations from lab frame to pixel coordinates.
    
    
    """
    if verbose: print('Converting translations from lab frame to pixel coordinates:\n')
    
    # transform the translations onto the ss / fs axes
    # map the translations onto the fs / ss axes
    dx_D = np.array([np.dot(basis[i], translations[i]) for i in range(len(basis))])
    dx_D[:, 0] /= x_pixel_size
    dx_D[:, 1] /= y_pixel_size
    
    # offset the translations so that the centre position is at the centre of the object
    dx_D[:, 0] -= np.mean(dx_D[:, 0])
    dx_D[:, 1] -= np.mean(dx_D[:, 1])
    
    dij_n = dx_D[:, :2].copy()
    dij_n[:, 0] /= dx
    dij_n[:, 1] /= dy
    return dij_n
