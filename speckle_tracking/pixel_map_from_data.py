import numpy as np
import tqdm

from .make_object_map import make_object_map
from .update_pixel_map import update_pixel_map
from .update_translations import update_translations
from .calc_error import calc_error

# it would be great to generalise this so that 
# no initial estimate of the positions is required

def pixel_map_from_data_old(data, xy, W = None, mask = None, u = None, 
                        search_window = None, filter=1., maxiters=10, 
                        tol = 1e-3, rad = None):
    """
    solve :
        data[n,i,j] = O[u[0,i,j]-x[n], u[1,i,j]-y[n]]
    
    for u.
    
    Optionally provide W:
        data[n,i,j] = W[i,j] O[u[0,i,j]-x[n], u[1,i,j]-y[n]]
    
    It is assumed that u is irrotational: curl u = 0
    """
    shape = data.shape
    
    if mask is None :
        mask = np.ones(shape[1:], dtype=np.bool)
    
    if u is None :
        u = np.array(np.indices(shape[1:])).astype(np.float)
    
    if W is None :
        W = np.ones(shape[1:], dtype=np.float64)
    
    # update the pixel translations
    xy, res = update_translations(data, mask, W, None, u, None, None, xy, 
                                  update_O=True, roi = [20, 80, 20, 80])
    
    if rad is not None :
        rads = np.linspace(rad,  
                              np.sqrt(W.shape[0]**2+W.shape[1]**2)/2, maxiters) 
        ss, fs = np.indices(shape[1:]).astype(np.float)
        ss -= shape[1]//2
        fs -= shape[2]//2
    
    u0      = u.copy()
    us = [u.copy()]
    
    # reconstruct with a radially expanding mask
    it = tqdm.trange(maxiters, desc='updating pixel and object maps')
    for j in it:
        
        if rad is not None :
            mask_circ = (ss**2 + fs**2) < rads[j]**2
            mask_circ *= mask
        else :
            mask_circ = mask
         
        O, n0, m0 = make_object_map(data, mask_circ, W, xy, u, subpixel=True)
            
        u, res = update_pixel_map(
                    data, mask_circ, W, O, u, n0, m0, xy, 
                    clip = [-40, 40],
                    fill_bad_pix = False, 
                    integrate = False, 
                    quadratic_refinement = True,
                    filter = None,
                    search_window = search_window)
        us.append(u.copy())
        #exp = exp*(u-u0) + u0
        
        uerr_new = res['error'] 
        if j>0 :
            if (uerr_old - uerr_new)/uerr_old < tol :
                it.close()
                break
        uerr_old = uerr_new
        
        des = "updating pixel and object maps: {:.2e}".format(res['error'])
        it.set_description(des)
    
    # reconstruct and gradually reduce filter factor, with integrate=True
    filters = np.linspace(filter, 0, maxiters)
    it = tqdm.trange(maxiters, desc='updating pixel and object maps')
    for j in it:
        data = flux_correction(data, W, O, n0, m0, u, xy, mask)
        
        O, n0, m0 = make_object_map(data, mask, W, xy, u, subpixel=True)
            
        u, res = update_pixel_map(
                    data, mask, W, O, u, n0, m0, xy, 
                    clip = [-40, 40],
                    fill_bad_pix = False, 
                    integrate = True, 
                    quadratic_refinement = True,
                    filter = filters[j],
                    search_window = search_window)
        us.append(u.copy())
        
        uerr_new = res['error'] 
        if j>0 :
            if (uerr_old - uerr_new)/uerr_old < tol :
                it.close()
                break
        uerr_old = uerr_new
        
        des = "updating pixel and object maps: {:.2e}".format(res['error'])
        it.set_description(des)
    
    """
    # first update O and xy until convergence
    it = tqdm.trange(maxiters, desc='updating object map and translations')
    for i in it:
        O, n0, m0 = make_object_map(data, mask, W, xy, u, subpixel=True)
        xy, res   = update_translations(data, mask, W, O, u, n0, m0, xy)
        err_new   = res['error']
        
        if i>0 and (err_old - err_new)/err_old < tol :
            break
        err_old = err_new
        it.set_description("updating object map and translations: {:.2e}".format(res['error']))
    """
    
    # first update O and xy until convergence
    it = tqdm.trange(maxiters, desc='updating pixel, object and translations')
    for j in it:
        u, res = update_pixel_map(
                    data, mask, W, O, u, n0, m0, xy, 
                    clip = [-40, 40],
                    fill_bad_pix = True, 
                    integrate = True, 
                    quadratic_refinement = True,
                    search_window = search_window)        
        us.append(u.copy())
        
        uerr_new = res['error'] 
        if j>0 and (uerr_old - uerr_new)/uerr_old < tol :
            break
        uerr_old = uerr_new
        
        it.set_description("updating pixel, object and translations: {:.2e}".format(res['error']))
        
        O, n0, m0 = make_object_map(data, mask, W, xy, u, subpixel=True)
        xy, res   = update_translations(data, mask, W, O, u, n0, m0, xy)
        
    return u, {'object_map': O, 'n0': n0, 'm0': m0, 'pix_positions': xy, 
               'm': mask_circ, 'us': us}

def pixel_map_from_data(data, xy, W = None, mask = None, u = None, 
                        search_window = None, filter=1., maxiters=10, 
                        tol = 1e-3, rad = None):
    """
    solve :
        data[n,i,j] = O[u[0,i,j]-x[n], u[1,i,j]-y[n]]
    
    for u.
    
    Optionally provide W:
        data[n,i,j] = W[i,j] O[u[0,i,j]-x[n], u[1,i,j]-y[n]]
    
    It is assumed that u is irrotational: curl u = 0
    """
    shape = data.shape
    
    if mask is None :
        mask = np.ones(shape[1:], dtype=np.bool)
    
    if u is None :
        u = np.array(np.indices(shape[1:])).astype(np.float)
    
    if W is None :
        W = np.ones(shape[1:], dtype=np.float64)
    
    # update the pixel translations
    xy, res = update_translations(data, mask, W, None, u, None, None, xy, 
                                  update_O=True, roi = [20, 80, 20, 80])
    
    O, n0, m0 = make_object_map(data, mask, W, xy, u, subpixel=True)
    
    # first update O and xy until convergence
    us = []
    it = tqdm.trange(maxiters, desc='updating pixel, object and translations')
    for j in it:
        u, res = update_pixel_map(
                    data, mask, W, O, u, n0, m0, xy, 
                    clip = [-40, 40],
                    fill_bad_pix = True, 
                    integrate = True, 
                    quadratic_refinement = True,
                    search_window = search_window)        
        us.append(u.copy())
        
        uerr_new = res['error'] 
        if j>0 and (uerr_old - uerr_new)/uerr_old < tol :
            break
        uerr_old = uerr_new
        
        it.set_description("updating pixel, object and translations: {:.2e}".format(res['error']))
        
        O, n0, m0 = make_object_map(data, mask, W, xy, u, subpixel=True)
        #xy, res   = update_translations(data, mask, W, O, u, n0, m0, xy)
        
    return u, {'object_map': O, 'n0': n0, 'm0': m0, 'pix_positions': xy, 
               'us': us}

from .calc_error import bilinear_interpolation_array
def flux_correction(data, W, O, n0, m0, u1, dij_n, mask):
    cs    = [] 
    for n in tqdm.trange(data.shape[0], desc='calculating errors'):
        # define the coordinate mapping and round to int
        ss = u1[0] - dij_n[n, 0] + n0
        fs = u1[1] - dij_n[n, 1] + m0
        #
        I0 = W * bilinear_interpolation_array(O, ss, fs, fill=-1, invalid=-1)
        #
        m = I0>0
        
        c = np.sum(mask * m * I0 * data[n]) / np.sum(mask * m * data[n]**2)
        cs.append(c)
    return (data.T * np.array(cs)).T.astype(np.float32)
