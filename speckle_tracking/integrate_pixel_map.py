from .utils import integrate
import numpy as np

def integrate_pixel_map(pixel_map, weight, wavlength, z_sample_dist, maxiter=3000):
    """
    iu - dx (pixel_map[i, j] + n0) = wav z / 2 pi grad(i u, j v) 
    
    grad_ss(iu, jv) = 2 pi / (z wav) ( i u - dx (pixel_map[0][i, j] + n0) )
    grad_fs(iu, jv) = 2 pi / (z wav) ( j v - dx (pixel_map[1][i, j] + m0) )
    
    phase(i u, j v) = u \sum_i grad_ss(i u, j v)
                    = v \sum_j grad_fs(i u, j v)
    """
    ij = np.indices(pixel_map.shape[1:])
    grad[0] = ij[0] * x_pixel_size - dxy[0] * (pixel_map[0] + n0)
    grad[1] = ij[1] * y_pixel_size - dxy[1] * (pixel_map[1] + n0)
    phase, res = integrate(
                 pixel_map[0], pixel_map[1], 
                 weight, maxiter, step=[x_pixel_size, y_pixel_size])
    phase *= 2. * np.pi / (wavlength * z_sample_dist) 
    return phase, {'pixel_map': np.array([res['dss_forward'], res['dfs_forward']])}
