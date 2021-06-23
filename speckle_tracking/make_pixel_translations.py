import numpy as np

def make_pixel_translations(translations, basis, dx, dy, verbose=False):
    """
    Convert sample translations from lab frame to pixel coordinates.
    
    
    """
    if verbose: print('Converting translations from lab frame to pixel coordinates:\n')
    
    x_pixel_size = np.sqrt(np.sum(basis[0, 0]**2))
    y_pixel_size = np.sqrt(np.sum(basis[0, 1]**2))
    
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
