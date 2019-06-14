from .utils import integrate
import numpy as np

def integrate_pixel_map(pixel_map, weight, wavlength, z_sample_dist, maxiter=3000):
    phase, res = integrate(
                 pixel_map[0], pixel_map[1], 
                 weight, maxiter)
    phase *= 2. * np.pi / (wavlength * z_sample_dist) 
    return phase, {'pixel_map': np.array([res['dss_forward'], res['dfs_forward']])}
