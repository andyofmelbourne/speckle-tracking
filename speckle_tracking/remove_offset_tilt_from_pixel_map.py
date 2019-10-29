import numpy as np

def remove_offset_tilt_from_pixel_map(u):
    hp = np.mean(u[0], axis=1)
    vp = np.mean(u[1], axis=0)
    
    x    = np.arange(hp.shape[0])
    hfit = np.polyfit(x, hp, 1)
    
    y    = np.arange(vp.shape[0])
    vfit = np.polyfit(y, vp, 1)
    
    uout = u.copy()
    uout[0] -= x[:, np.newaxis] * hfit[0] + hfit[1]
    uout[1] -= y[np.newaxis, :] * vfit[0] + vfit[1]
    return uout
